from dataclasses import asdict, dataclass, field
from os import getenv
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.exceptions import EvalError
from agno.models.base import Model
from agno.utils.log import logger, set_log_level_to_debug, set_log_level_to_info

if TYPE_CHECKING:
    from rich.console import Console


class AccuracyAgentResponse(BaseModel):
    accuracy_score: int = Field(..., description="Accuracy Score between 1 and 10 assigned to the Agent's answer.")
    accuracy_reason: str = Field(..., description="Detailed reasoning for the accuracy score.")


@dataclass
class AccuracyEvaluation:
    prompt: str
    answer: str
    expected_answer: str
    score: int
    reason: str

    def print_eval(self, console: Optional["Console"] = None):
        from rich.box import ROUNDED
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.table import Table

        if console is None:
            console = Console()

        results_table = Table(
            box=ROUNDED,
            border_style="blue",
            show_header=False,
            title="[ Evaluation Result ]",
            title_style="bold sky_blue1",
            title_justify="center",
        )
        results_table.add_row("Prompt", self.prompt)
        results_table.add_row("Answer", self.answer)
        results_table.add_row("Expected Answer", self.expected_answer)
        results_table.add_row("Accuracy Score", f"{str(self.score)}/10")
        results_table.add_row("Accuracy Reason", Markdown(self.reason))
        console.print(results_table)


@dataclass
class AccuracyResult:
    results: List[AccuracyEvaluation] = field(default_factory=list)
    avg_score: float = field(init=False)
    mean_score: float = field(init=False)
    min_score: float = field(init=False)
    max_score: float = field(init=False)
    std_dev_score: float = field(init=False)

    def __post_init__(self):
        self.compute_stats()

    def compute_stats(self):
        import statistics

        if self.results and len(self.results) > 0:
            _results = [r.score for r in self.results]
            self.avg_score = statistics.mean(_results)
            self.mean_score = statistics.mean(_results)
            self.min_score = min(_results)
            self.max_score = max(_results)
            self.std_dev_score = statistics.stdev(_results) if len(_results) > 1 else 0

    def print_summary(self, console: Optional["Console"] = None):
        from rich.box import ROUNDED
        from rich.console import Console
        from rich.table import Table

        if console is None:
            console = Console()

        summary_table = Table(
            box=ROUNDED,
            border_style="blue",
            show_header=False,
            title="[ Evaluation Summary ]",
            title_style="bold sky_blue1",
            title_justify="center",
        )
        summary_table.add_row("Number of Runs", f"{len(self.results)}")
        summary_table.add_row("Average Score", f"{self.avg_score:.2f}")
        summary_table.add_row("Mean Score", f"{self.mean_score:.2f}")
        summary_table.add_row("Minimum Score", f"{self.min_score:.2f}")
        summary_table.add_row("Maximum Score", f"{self.max_score:.2f}")
        summary_table.add_row("Standard Deviation", f"{self.std_dev_score:.2f}")
        console.print(summary_table)

    def print_results(self, console: Optional["Console"] = None):
        from rich.box import ROUNDED
        from rich.console import Console
        from rich.table import Table

        if console is None:
            console = Console()

        results_table = Table(
            box=ROUNDED,
            border_style="blue",
            show_header=False,
            title="[ Evaluation Result ]",
            title_style="bold sky_blue1",
            title_justify="center",
        )
        for result in self.results:
            results_table.add_row("Prompt", result.prompt)
            results_table.add_row("Answer", result.answer)
            results_table.add_row("Expected Answer", result.expected_answer)
            results_table.add_row("Accuracy Score", f"{str(result.score)}/10")
            results_table.add_row("Accuracy Reason", result.reason)
        console.print(results_table)


@dataclass
class AccuracyEval:
    """Interface to evaluate the accuracy of an Agent, given a prompt and expected answer"""

    # Agent used in the evaluation
    agent: Agent
    # Prompt used in the evaluation
    prompt: Union[str, Callable]
    # The agent's expected answer to the given prompt
    expected_answer: Union[str, Callable]

    # Evaluation name
    name: Optional[str] = None
    # Evaluation UUID
    eval_id: str = field(default_factory=lambda: str(uuid4()))
    # Number of iterations to run
    num_iterations: int = 3
    # Result of the evaluation
    result: Optional[AccuracyResult] = None

    # Agent used to evaluate the answer
    evaluator_agent: Optional[Agent] = None
    # Guidelines for the evaluator agent
    evaluator_guidelines: Optional[List[str]] = None
    # Additional context to the evaluator agent
    evaluator_context: Optional[str] = None
    # Used to build the evaluator agent if not provided
    evaluator_model: Optional[Model] = None

    # Print summary of results
    print_summary: bool = False
    # Print detailed results
    print_results: bool = False
    # If set, results will be saved in the given file path
    file_path_to_save_results: Optional[str] = None
    # Enable debug logs
    debug_mode: bool = getenv("AGNO_DEBUG", "false").lower() == "true"

    def get_evaluator_agent(self) -> Agent:
        """Return the evaluator agent. If not provided, build it based on the evaluator fields and default instructions."""
        if self.evaluator_agent is not None:
            return self.evaluator_agent

        model = self.evaluator_model
        if model is None:
            try:
                from agno.models.openai import OpenAIChat

                model = OpenAIChat(id="gpt-4o-mini")
            except (ModuleNotFoundError, ImportError) as e:
                logger.exception(e)
                raise EvalError(
                    "Agno uses `openai` as the default model provider. Please run `pip install openai` to use the default evaluator."
                )

        evaluator_guidelines = ""
        if self.evaluator_guidelines is not None and len(self.evaluator_guidelines) > 0:
            evaluator_guidelines = "\n## Guidelines for the Agent's answer:\n"
            evaluator_guidelines += "\n- ".join(self.evaluator_guidelines)
            evaluator_guidelines += "\n"

        evaluator_context = ""
        if self.evaluator_context is not None and len(self.evaluator_context) > 0:
            evaluator_context = "## Additional Context:\n"
            evaluator_context += self.evaluator_context
            evaluator_context += "\n"

        return Agent(
            model=model,
            description=f"""\
You are an Agent Evaluator tasked with assessing the accuracy of an AI Agent's answer compared to an expected answer for a given question.
Your task is to provide a detailed analysis and assign a score on a scale of 1 to 10, where 10 indicates a perfect match to the expected answer.

## Prompt:
{self.prompt}

## Expected Answer:
{self.expected_answer}

## Evaluation Criteria:
1. Accuracy of information
2. Completeness of the answer
3. Relevance to the prompt
4. Use of key concepts and ideas
5. Overall structure and clarity of presentation
{evaluator_guidelines}{evaluator_context}
## Instructions:
1. Carefully compare the AI Agent's answer to the expected answer.
2. Provide a detailed analysis, highlighting:
   - Specific similarities and differences
   - Key points included or missed
   - Any inaccuracies or misconceptions
3. Explicitly reference the evaluation criteria and any provided guidelines in your reasoning.
4. Assign a score from 1 to 10 (use only whole numbers) based on the following scale:
   1-2: Completely incorrect or irrelevant
   3-4: Major inaccuracies or missing crucial information
   5-6: Partially correct, but with significant omissions or errors
   7-8: Mostly accurate and complete, with minor issues
   9-10: Highly accurate and complete, matching the expected answer closely

Your evaluation should be objective, thorough, and well-reasoned. Provide specific examples from both answers to support your assessment.""",
            response_model=AccuracyAgentResponse,
            structured_outputs=True,
        )

    def get_eval_expected_answer(self) -> str:
        """Return the eval expected answer. If it is a callable, call it and return the resulting string"""
        if callable(self.expected_answer):
            _answer = self.expected_answer()
            if isinstance(_answer, str):
                return _answer
            else:
                raise EvalError(f"The expected answer needs to be or return a string, but it returned: {type(_answer)}")
        return self.expected_answer

    def get_eval_prompt(self) -> str:
        """Return the eval prompt. If it is a callable, call it and return the resulting string"""
        if callable(self.prompt):
            _prompt = self.prompt()
            if isinstance(_prompt, str):
                return _prompt
            else:
                raise EvalError(f"The eval prompt needs to be or return a string, but it returned: {type(_prompt)}")
        return self.prompt

    def run(
        self,
        *,
        print_summary: bool = True,
        print_results: bool = True,
    ) -> Optional[AccuracyResult]:
        from rich.console import Console
        from rich.live import Live
        from rich.status import Status

        self.print_results = print_results
        self.print_summary = print_summary
        self.result = AccuracyResult()
        set_log_level_to_debug() if self.debug_mode else set_log_level_to_info()

        eval_prompt = self.get_eval_prompt()
        eval_expected_answer = self.get_eval_expected_answer()

        logger.debug(f"************ Evaluation Start: {self.eval_id} ************")
        logger.debug(f"Evaluation Prompt: {eval_prompt}")
        logger.debug(f"Evaluation Expected Answer: {eval_expected_answer}")
        logger.debug("***********************************************************")

        evaluator_agent = self.get_evaluator_agent()

        # Add a spinner while running the evaluations
        console = Console()
        with Live(console=console, transient=True) as live_log:
            for i in range(self.num_iterations):
                status = Status(f"Running evaluation {i + 1}...", spinner="dots", speed=1.0, refresh_per_second=10)
                live_log.update(status)

                answer = self.agent.run(message=eval_prompt).content
                if not answer:
                    logger.error(f"Failed to generate a valid answer on iteration {i + 1}: {answer}")
                    continue

                try:
                    logger.debug(f"Answer #{i + 1}: {answer}")
                    accuracy_agent_response = evaluator_agent.run(answer).content
                    if accuracy_agent_response is None or not isinstance(
                        accuracy_agent_response, AccuracyAgentResponse
                    ):
                        logger.error(f"Evaluator Agent returned an invalid response on iteration {i + 1}")
                        continue
                    accuracy_evaluation = AccuracyEvaluation(
                        prompt=eval_prompt,
                        answer=answer,  # type: ignore
                        expected_answer=eval_expected_answer,
                        score=accuracy_agent_response.accuracy_score,
                        reason=accuracy_agent_response.accuracy_reason,
                    )
                    if self.print_results:
                        accuracy_evaluation.print_eval(console)
                    self.result.results.append(accuracy_evaluation)
                    self.result.compute_stats()
                    status.update(f"Eval iteration {i + 1} finished")
                except Exception as e:
                    logger.exception(f"Failed to evaluate accuracy, run #{i + 1}: {e}")
                    return None
                status.stop()

        # Save result to file if requested
        if self.file_path_to_save_results is not None and self.result is not None:
            try:
                import json

                fn_path = Path(self.file_path_to_save_results.format(name=self.name, eval_id=self.eval_id))
                if not fn_path.parent.exists():
                    fn_path.parent.mkdir(parents=True, exist_ok=True)
                fn_path.write_text(json.dumps(asdict(self.result), indent=4))
            except Exception as e:
                logger.warning(f"Failed to save eval results to file {self.file_path_to_save_results}: {e}")

        # Show results if requested
        if self.print_results:
            self.result.print_results(console)
        elif self.print_summary:
            self.result.print_summary(console)

        logger.debug(f"*********** Evaluation {self.eval_id} Finished ***********")
        return self.result

    def run_with_given_answer(
        self,
        *,
        answer: str,
        print_summary: bool = True,
        print_results: bool = True,
    ) -> Optional[AccuracyResult]:
        """Run the evaluation logic against the given answer, instead of generating an answer with the Agent"""
        from rich.console import Console

        self.print_results = print_results
        self.print_summary = print_summary
        self.result = AccuracyResult()
        set_log_level_to_debug() if self.debug_mode else set_log_level_to_info()

        eval_prompt = self.get_eval_prompt()
        eval_expected_answer = self.get_eval_expected_answer()

        logger.debug(f"************ Evaluation Start: {self.eval_id} ************")
        logger.debug(f"Evaluation Prompt: {eval_prompt}")
        logger.debug(f"Evaluation Answer: {answer}")
        logger.debug(f"Evaluation Expected Answer: {eval_expected_answer}")
        logger.debug("***********************************************************")

        evaluator_agent = self.get_evaluator_agent()

        try:
            accuracy_agent_response = evaluator_agent.run(answer).content
            if accuracy_agent_response is None or not isinstance(accuracy_agent_response, AccuracyAgentResponse):
                raise EvalError(f"Evaluator Agent returned an invalid response: {accuracy_agent_response}")
            accuracy_evaluation = AccuracyEvaluation(
                prompt=eval_prompt,
                answer=answer,  # type: ignore
                expected_answer=eval_expected_answer,
                score=accuracy_agent_response.accuracy_score,
                reason=accuracy_agent_response.accuracy_reason,
            )
            if self.print_results:
                console = Console()
                accuracy_evaluation.print_eval(console)
            self.result.results.append(accuracy_evaluation)
            self.result.compute_stats()
        except Exception as e:
            logger.exception(f"Failed to evaluate accuracy: {e}")
            return None

        # Save result to file if requested
        if self.file_path_to_save_results is not None and self.result is not None:
            try:
                import json

                fn_path = Path(self.file_path_to_save_results.format(name=self.name, eval_id=self.eval_id))
                if not fn_path.parent.exists():
                    fn_path.parent.mkdir(parents=True, exist_ok=True)
                fn_path.write_text(json.dumps(asdict(self.result), indent=4))
            except Exception as e:
                logger.warning(f"Failed to save eval results to file {self.file_path_to_save_results}: {e}")

        # Show results if requested
        if self.print_results:
            self.result.print_results(console)
        elif self.print_summary:
            self.result.print_summary(console)

        logger.debug(f"*********** Evaluation {self.eval_id} Finished ***********")
        return self.result
