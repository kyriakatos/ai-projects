import argparse
from agents import Runner

from .agent import build_agent
from .logging_config import setup_logging

def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Daily Sentiment Agent CLI")
    parser.add_argument("prompt", nargs="*", help='e.g. "Fetch last 5 years from stocktwits into outputs"')
    args = parser.parse_args()

    prompt = " ".join(args.prompt).strip() or "Fetch last 5 years daily sentiment from stocktwits into outputs."
    agent = build_agent()
    result = Runner.run_sync(agent, prompt)
    print(result.final_output)