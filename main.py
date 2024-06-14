import sys
from datetime import datetime
from dotenv import load_dotenv
from unstract.llmwhisperer.client import LLMWhispererClient, LLMWhispererClientException
from pydantic import BaseModel, Field
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser


# define a Pydantic schema for credit card spends with purchase date,
# merchant name and amount spent
class CreditCardSpend(BaseModel):
    spend_date: datetime = Field(description="Date of purchase")
    merchant_name: str = Field(description="Name of the merchant")
    amount_spent: float = Field(description="Amount spent")


class CreditCardSpendItems(BaseModel):
    spend_items: list[CreditCardSpend] = Field(description="List of spend items from the credit card statement")


class RegionalFinancialStatement(BaseModel):
    quarter_ending: datetime = Field(description="Quarter ending date")
    net_sales: float = Field(description="Net sales")
    operating_income: float = Field(description="Operating income")
    ending_type: str = Field(description="Type of ending. Set to either '6-month' or '3-month'")


class GeographicFinancialStatement(BaseModel):
    americas: list[RegionalFinancialStatement] = Field(description="Financial statement for the Americas region, "
                                                                   "sorted chronologically")
    europe: list[RegionalFinancialStatement] = Field(description="Financial statement for the Europe region, sorted "
                                                                 "chronologically")
    greater_china: list[RegionalFinancialStatement] = Field(description="Financial statement for the Greater China "
                                                                        "region, sorted chronologically")
    japan: list[RegionalFinancialStatement] = Field(description="Financial statement for the Japan region, sorted "
                                                                "chronologically")
    rest_of_asia_pacific: list[RegionalFinancialStatement] = Field(description="Financial statement for the Rest of "
                                                                               "Asia Pacific region, sorted "
                                                                               "chronologically")


def error_exit(error_message):
    print(error_message)
    sys.exit(1)


def extract_text_from_pdf(file_path, pages_list=None):
    llmw = LLMWhispererClient()
    try:
        result = llmw.whisper(file_path=file_path, pages_to_extract=pages_list)
        extracted_text = result["extracted_text"]
        return extracted_text
    except LLMWhispererClientException as e:
        error_exit(e)


def compile_template_and_get_llm_response(preamble, extracted_text, pydantic_object):
    postamble = "Do not include any explanation in the reply. Only include the extracted information in the reply."
    system_template = "{preamble}"
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_template = "{format_instructions}\n\n{extracted_text}\n\n{postamble}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    parser = PydanticOutputParser(pydantic_object=pydantic_object)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    request = chat_prompt.format_prompt(preamble=preamble,
                                        format_instructions=parser.get_format_instructions(),
                                        extracted_text=extracted_text,
                                        postamble=postamble).to_messages()
    chat = ChatOpenAI()
    response = chat(request, temperature=0.0)
    print(f"Response from LLM:\n{response.content}")
    return response.content


def extract_cc_spend_from_text(extracted_text):
    preamble = ("You're seeing the list of spend items from a credit card statement and your job is to accurately "
                "extract the spend date, merchant name and amount spent for each transaction.")
    return compile_template_and_get_llm_response(preamble, extracted_text, CreditCardSpendItems)


def process_cc_statement():
    extracted_text = extract_text_from_pdf("assets/docs/Chase Freedom.pdf", pages_list="3")
    print(extracted_text)
    response = extract_cc_spend_from_text(extracted_text)
    print(response)


def extract_financial_statement_from_text(extracted_text):
    preamble = ("You're seeing the financial statement for a company and your job is to accurately extract the "
                "revenue, cost of revenue, gross profit, operating income, net income and earnings per share.")
    return compile_template_and_get_llm_response(preamble, extracted_text, GeographicFinancialStatement)


def process_financial_statement():
    extracted_text = extract_text_from_pdf("assets/docs/Apple_10-Q-Q2-2024.pdf", pages_list="14")
    print(extracted_text)
    response = extract_financial_statement_from_text(extracted_text)
    print(response)


def main():
    load_dotenv()
    process_cc_statement()
    process_financial_statement()


if __name__ == "__main__":
    main()
