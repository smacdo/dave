import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def main() -> None:
    load_dotenv()

    # Prompt template.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that provides concise answers."),
            ("user", "{input}"),
        ]
    )

    # Load local LLM with the OpenAI compatible client.
    llm = ChatOpenAI(
        base_url=os.environ["OPENAI_API_BASE"],
        api_key=os.environ["OPENAI_API_KEY"],
        model="local-model",
        temperature=0.7,  # Adjust temperature for creativity
    )

    # Chain together.
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    # Invoke the chain
    question = "Explain the difference between an 'apple' and a 'bobcat'."
    response = chain.invoke({"input": question})

    print(response)
    print("---DONE---")


if __name__ == "__main__":
    main()