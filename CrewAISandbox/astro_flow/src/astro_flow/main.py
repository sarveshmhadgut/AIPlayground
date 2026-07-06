from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from astro_flow.crews.content_crew.content_crew import ContentCrew
from astro_flow.crews.publish_crew.publish_crew import PublishCrew


class ContentState(BaseModel):
    topic: str = ""
    outline: str = ""
    draft: str = ""
    final_post: str = ""
    published_post: str = ""


class ContentFlow(Flow[ContentState]):
    @start()
    def plan_content(self, crewai_trigger_payload: dict = None):
        print("Planning content")

        if crewai_trigger_payload:
            self.state.topic = crewai_trigger_payload.get("topic", "Artimis II Mission")
            print(f"Using trigger payload: {crewai_trigger_payload}")
        else:
            self.state.topic = "Artimis II Mission"

        print(f"Topic: {self.state.topic}")

    @listen(plan_content)
    def generate_content(self):
        print(f"Generating content on: {self.state.topic}")
        result = (
            ContentCrew().crew().kickoff(inputs={"topic": self.state.topic, "current_year": str(datetime.now().year)})
        )

        print("Content generated")
        self.state.final_post = result.raw

    @listen(generate_content)
    def publish_content(self):
        print(f"Publishing content for: {self.state.topic}")
        result = PublishCrew().crew().kickoff(inputs={"topic": self.state.topic, "draft_report": self.state.final_post})

        print("Content published")
        self.state.published_post = result.raw

    @listen(publish_content)
    def save_content(self):
        print("Saving published content")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        with open(output_dir / "post.md", "w") as f:
            f.write(self.state.published_post)
        print("Post saved to output/post.md")


def kickoff():
    content_flow = ContentFlow()
    content_flow.kickoff()


def plot():
    content_flow = ContentFlow()
    content_flow.plot()


def run_with_trigger():
    """
    Run the flow with trigger payload.
    """
    import json
    import sys

    # Get trigger payload from command line argument
    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    # Create flow and kickoff with trigger payload
    # The @start() methods will automatically receive crewai_trigger_payload parameter
    content_flow = ContentFlow()

    try:
        result = content_flow.kickoff({"crewai_trigger_payload": trigger_payload})
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the flow with trigger: {e}")


if __name__ == "__main__":
    kickoff()
