import argparse
import logging
from agents import Runner  # Agents SDK Runner  [oai_citation:4‡openai.github.io](https://openai.github.io/openai-agents-python)

from .agent import build_agent
from .logging_config import setup_logging

log = logging.getLogger("stock_agent.cli")

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Stock Agent CLI")
    parser.add_argument(
        "prompt",
        nargs="*",
        help='Natural language prompt, e.g. "Fetch last 5 years for Apple, NVIDIA into outputs/"',
    )
    args = parser.parse_args()

    prompt = " ".join(args.prompt).strip() or "Fetch last 5 years for Apple, Google, Meta, NVIDIA, Oracle."
    agent = build_agent()

    result = Runner.run_sync(agent, prompt)
    print(result.final_output)


## Run locally in bash
## export OPENAI_API_KEY="..."
## stock-agent "Fetch last 5 years and write into outputs/"
##