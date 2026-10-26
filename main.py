"""Command-line interface for the BD Legal Research Assistant.

Provides interactive execution of the deep_agent graph with terminal-based
human-in-the-loop analyst review, streaming progress, and multi-format exports.
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from src.deep_agent import builder as deep_agent_builder
from src.utils.export import export_to_json

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("research_assistant_cli")


def run_research(topic: str, max_analysts: int = 3, turns: int = 2, output_file: str = "report.md", json_file: str = "report.json"):
    print("\n" + "=" * 60)
    print("🏛️  BANGLADESH LEGAL RESEARCH ASSISTANT (LangGraph)")
    print("=" * 60)
    print(f"Topic: {topic}")
    print(f"Analysts requested: {max_analysts} | Max interview turns: {turns}")
    print("=" * 60 + "\n")

    checkpointer = MemorySaver()
    app = deep_agent_builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "cli-research-session"}}

    initial_state = {
        "topic": topic,
        "max_analysts": max_analysts,
        "interview_turns": turns,
        "sections": [],
    }

    print("⏳ Step 1: Generating legal analyst personas...")
    events = app.stream(initial_state, config=config)

    for event in events:
        for node_name, state_update in event.items():
            print(f"  ✓ Completed node: {node_name}")

    # Check for human-in-the-loop pause
    snapshot = app.get_state(config)
    while snapshot.next and "human_feedback" in snapshot.next:
        # We are paused at human_feedback interrupt
        interrupts = snapshot.tasks[0].interrupts if snapshot.tasks else []
        payload = interrupts[0].value if interrupts else {}

        print("\n" + "-" * 50)
        print("👤 HUMAN REVIEW: Generated Legal Analysts")
        print("-" * 50)
        analysts = payload.get("analysts", [])
        for idx, a in enumerate(analysts, 1):
            name = a.get("name", "Analyst")
            role = a.get("role", "Legal Researcher")
            domain = a.get("legal_domain", "Bangladesh Law")
            print(f"  [{idx}] {name} — {role} ({domain})")

        print("-" * 50)
        user_input = input("Press [Enter] to approve, or type feedback to revise: ").strip()

        print("\n⏳ Resuming research pipeline with your response...")
        events = app.stream(Command(resume=user_input or "approved"), config=config)
        for event in events:
            for node_name, state_update in event.items():
                print(f"  ✓ Completed node: {node_name}")
        snapshot = app.get_state(config)

    final_state = snapshot.values
    final_report = final_state.get("final_report", "No report was generated.")

    # Save markdown report
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_report)
    print(f"\n📄 Saved Markdown report to: {output_file}")

    # Save structured JSON export
    try:
        json_output = export_to_json(final_state)
        with open(json_file, "w", encoding="utf-8") as f:
            f.write(json_output)
        print(f"📊 Saved Structured JSON to: {json_file}")
    except Exception as e:
        logger.warning("Could not export JSON report: %s", e)

    print("\n✅ Research completed successfully!\n")


def main():
    parser = argparse.ArgumentParser(description="Bangladesh Legal Research Assistant CLI")
    parser.add_argument("--topic", "-t", type=str, help="Legal research topic or question")
    parser.add_argument("--analysts", "-a", type=int, default=3, help="Maximum analyst personas (default: 3)")
    parser.add_argument("--turns", type=int, default=2, help="Maximum interview turns per analyst (default: 2)")
    parser.add_argument("--output", "-o", type=str, default="report.md", help="Markdown output path (default: report.md)")
    parser.add_argument("--json", "-j", type=str, default="report.json", help="JSON export path (default: report.json)")

    args = parser.parse_args()

    topic = args.topic
    if not topic:
        try:
            topic = input("Enter Bangladesh legal research topic: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(0)

    if not topic:
        print("Error: topic cannot be empty.")
        sys.exit(1)

    run_research(
        topic=topic,
        max_analysts=args.analysts,
        turns=args.turns,
        output_file=args.output,
        json_file=args.json,
    )


if __name__ == "__main__":
    main()
