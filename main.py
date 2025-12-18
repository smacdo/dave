import os

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

SYSTEM_PROMPT_TEMPLATE = """
    Brainstorm a list names for a new coding project or library. The name should be easy to remember,
    memorable, and relevant. The name should consistent of only letters, numbers, and hyphens.

    Check that your suggestions are not taken before adding them to the list of results. If the is
    not taken (false), add it to the list. If it is taken (true), then try coming up with another
    name. Do not check more than twenty names before giving up and returning `"NO NAMES GENERATED"`.
    
    Return the results as a JSON object with the following format:
    [
       { name: "${PACKAGE_NAME}", reason: "${REASON_FOR_SUGGESTION}" }, 
    ]
"""


@tool
def is_package_taken(name):
    """
    Search PyPI to see if the Python package name is taken.

    :param name: The name of the package to check.
    :return: True if the package name is taken, False if the name is available.
    """
    url = f"https://pypi.org/pypi/{name}/json"
    response = requests.get(url)
    return response.status_code == 200


@tool
def is_crate_taken(name):
    """
    Search crates.io to see if the Rust package name is taken.

    :param name: The name of the package to check.
    :return: True if the package name is taken, False if the name is available.
    """
    url = f"https://crates.io/api/v1/crates/{name}"
    headers = {"User-Agent": "package-checker"}  # crates.io requires a User-Agent
    response = requests.get(url, headers=headers)
    return response.status_code == 200


@tool
def is_npm_package_taken(name):
    """
    Search NPM to see if the JavaScript package name is taken.

    :param name: The name of the package to check.
    :return: True if the package name is taken, False if the name is available.
    """
    url = f"https://registry.npmjs.org/{name}"
    response = requests.get(url)
    return response.status_code == 200


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
        model="mistralai/devstral-small-2-2512",
        temperature=0.7,  # Adjust temperature for creativity
    )

    # Create an agent.
    agent = create_agent(
        llm,
        tools=[is_package_taken, is_crate_taken, is_npm_package_taken],
        system_prompt=SYSTEM_PROMPT_TEMPLATE,
    )

    # Invoke the agent.
    print("---SEND TO AI---")

    project_type = "library"
    project_lang = "python"
    project_desc = "helps developers come up with new names for Python packages"

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"Top ten name suggestions for a {project_lang} {project_type} package that {project_desc}",
                }
            ]
        }
    )

    print("---RESPONSE BELOW---")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()