from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class PublishCrew:
    """Publish Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def seo_optimizer(self) -> Agent:
        return Agent(
            config=self.agents_config["seo_optimizer"],  # type: ignore[index]
        )

    @agent
    def compliance_checker(self) -> Agent:
        return Agent(
            config=self.agents_config["compliance_checker"],  # type: ignore[index]
        )

    @agent
    def formatter(self) -> Agent:
        return Agent(
            config=self.agents_config["formatter"],  # type: ignore[index]
        )

    @agent
    def performance_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["performance_analyst"],  # type: ignore[index]
        )

    @agent
    def publisher(self) -> Agent:
        return Agent(
            config=self.agents_config["publisher"],  # type: ignore[index]
        )

    @task
    def seo_optimization_task(self) -> Task:
        return Task(
            config=self.tasks_config["seo_optimization_task"],  # type: ignore[index]
        )

    @task
    def compliance_check_task(self) -> Task:
        return Task(
            config=self.tasks_config["compliance_check_task"],  # type: ignore[index]
        )

    @task
    def formatting_task(self) -> Task:
        return Task(
            config=self.tasks_config["formatting_task"],  # type: ignore[index]
        )

    @task
    def performance_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["performance_analysis_task"],  # type: ignore[index]
        )

    @task
    def publish_task(self) -> Task:
        return Task(
            config=self.tasks_config["publish_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Publish Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
