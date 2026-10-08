import unittest
import os
import pandas as pd
from unittest.mock import patch, MagicMock

import cais.agent as cais_agent
from cais.agent import CausalAgent
from cais.models import AssumptionResult, Variables
from cais.components.decision_tree_v2 import TreeResult, TreeStep


def create_dummy_csv(path='dummy_e2e_test_data.csv'):
    df = pd.DataFrame({
        'treatment': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'outcome': [10, 12, 11, 13, 9, 14, 10, 15, 11, 16],
        'covariate1': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        'covariate2': [5.5, 6.5, 5.8, 6.2, 5.1, 6.8, 5.3, 6.1, 5.9, 6.3],
    })
    df.to_csv(path, index=False)
    return path


class TestAgentWorkflow(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dummy_data_path = create_dummy_csv()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.dummy_data_path):
            os.remove(cls.dummy_data_path)

    @patch('cais.agent.run_causal_analysis')
    def test_agent_invocation_mock(self, mock_run):
        """Smoke test: run_causal_analysis can be called and returns a result."""
        mock_run.return_value = {"explanation": "Agent invoked successfully (mocked)", "results": {}}

        result = cais_agent.run_causal_analysis(
            "What is the effect of treatment on outcome?",
            self.dummy_data_path,
        )

        self.assertIsInstance(result, dict)
        self.assertIn("explanation", result)

    @patch('cais.agent.get_llm_client')
    def test_causal_agent_initialization(self, mock_get_llm):
        """CausalAgent initializes with correct attributes."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        agent = CausalAgent(
            dataset_path=self.dummy_data_path,
            dataset_description="A simple test dataset.",
        )

        # Core attributes should be set
        self.assertEqual(agent.dataset_path, self.dummy_data_path)
        self.assertEqual(agent.dataset_description, "A simple test dataset.")
        self.assertIs(agent.llm, mock_llm)
        self.assertIsNotNone(agent.estimators)

        # Pipeline states should start as None
        self.assertIsNone(agent.dataset_analysis)
        self.assertIsNone(agent.variables)
        self.assertIsNone(agent.selected_method)
        self.assertIsNone(agent.results)

    @patch('cais.agent.get_llm_client')
    def test_causal_agent_load_dataset(self, mock_get_llm):
        """CausalAgent.load_dataset() returns a DataFrame from the CSV path."""
        mock_get_llm.return_value = MagicMock()

        agent = CausalAgent(dataset_path=self.dummy_data_path)
        df = agent.load_dataset()

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("treatment", df.columns)
        self.assertIn("outcome", df.columns)
        self.assertEqual(len(df), 10)

    @patch('cais.agent.get_llm_client')
    def test_causal_agent_checkq_stores_and_retrieves_query(self, mock_get_llm):
        """CausalAgent.checkq() stores and later retrieves the last used query."""
        mock_get_llm.return_value = MagicMock()

        agent = CausalAgent(dataset_path=self.dummy_data_path)

        # First call stores the query
        returned = agent.checkq("What is the effect of X on Y?")
        self.assertEqual(returned, "What is the effect of X on Y?")
        self.assertEqual(agent.last_used_query, "What is the effect of X on Y?")

        # Calling with None falls back to the stored query
        returned_again = agent.checkq(None)
        self.assertEqual(returned_again, "What is the effect of X on Y?")


class TestDecisionTreeV2Wiring(unittest.TestCase):
    """Agent wiring for the revised tree: no LLM calls, tree stubbed."""

    @classmethod
    def setUpClass(cls):
        cls.dummy_data_path = create_dummy_csv('dummy_tree_v2_test_data.csv')

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.dummy_data_path):
            os.remove(cls.dummy_data_path)

    def _make_agent(self):
        with patch('cais.agent.get_llm_client', return_value=MagicMock()):
            agent = CausalAgent(
                dataset_path=self.dummy_data_path,
                dataset_description="A simple test dataset.",
            )
        agent.dataset_analysis = None
        agent.variables = Variables(
            treatment_variable="treatment",
            outcome_variable="outcome",
            treatment_variable_type="binary",
            covariates=["covariate1", "covariate2"],
            is_rct=False,
        )
        return agent

    def test_tree_properties_translation(self):
        agent = self._make_agent()
        agent.variables.confounders = ["covariate1", "extra_confounder"]

        props = agent._tree_properties()

        self.assertEqual(props["treatment_variable"], "treatment")
        self.assertEqual(props["outcome_variable"], "outcome")
        # confounders are merged into covariates without duplicates
        self.assertEqual(props["covariates"], ["covariate1", "covariate2", "extra_confounder"])
        self.assertTrue(props["has_valid_backdoor_set"])
        self.assertFalse(props["has_valid_instrument"])
        self.assertFalse(props["has_temporal_structure"])
        self.assertIsNone(props["mediator_variable"])
        self.assertIsNone(props["placebo_period_start"])

    @patch('cais.agent.run_decision_tree_v2')
    def test_select_method_v2_records_result(self, mock_tree):
        agent = self._make_agent()
        mock_tree.return_value = TreeResult(
            method="difference_in_differences",
            steps=[TreeStep("Gate: parallel trends", "gate", "pass")],
            assumptions={"parallel_trends": AssumptionResult(passed=True, reasoning="p=0.4")},
        )

        result = agent.select_method_v2()

        self.assertEqual(result.method, "difference_in_differences")
        self.assertEqual(agent.selected_method, "difference_in_differences")
        self.assertEqual(agent.method_info.selected_method, "difference_in_differences")
        self.assertEqual(agent.method_info.decision_tree["method"], "difference_in_differences")
        self.assertIn("Decision tree v2 selected", agent.method_info.method_justification)
        # the raw dataframe and description are handed to the tree
        kwargs = mock_tree.call_args.kwargs
        self.assertEqual(kwargs["description"], "A simple test dataset.")
        self.assertIsInstance(kwargs["df"], pd.DataFrame)

    def test_validation_info_shape_matches_explainer(self):
        agent = self._make_agent()
        agent.tree_result = TreeResult(
            method="propensity_score_matching",
            steps=[TreeStep("leaf", "leaf", "selected")],
            assumptions={"positivity": AssumptionResult(passed=True, reasoning="overlap ok", details={"n": 1})},
            warnings=["SUTVA could not be assessed (advisory)."],
            planned_post_checks=["balance_after_matching"],
        )

        info = agent._tree_validation_info()

        for key in ("method", "variant", "path", "steps", "assumptions", "concerns", "planned_post_checks"):
            self.assertIn(key, info)
        self.assertEqual(info["method"], "propensity_score_matching")
        self.assertEqual(info["assumptions"]["positivity"]["passed"], True)
        self.assertIn("SUTVA could not be assessed (advisory).", info["concerns"])
        self.assertEqual(info["planned_post_checks"], ["balance_after_matching"])

    def test_tree_stop_output_for_gate_failure(self):
        agent = self._make_agent()
        agent.tree_result = TreeResult(
            ended=True,
            end_reason="Parallel trends violated.",
            steps=[
                TreeStep("Gate: parallel trends", "gate", "fail", "p=0.01"),
                TreeStep("END", "end", "stop", "Parallel trends violated."),
            ],
            assumptions={"parallel_trends": AssumptionResult(passed=False, reasoning="p=0.01")},
        )

        output = agent._tree_stop_output()

        self.assertEqual(output["results"]["status"], "not_estimated")
        self.assertEqual(output["results"]["end_reason"], "Parallel trends violated.")
        self.assertIn("Parallel trends violated.", output["explanation"])
        self.assertIn("Gate: parallel trends: fail (p=0.01)", output["explanation"])
        self.assertIn("parallel_trends: fail - p=0.01", output["explanation"])
        self.assertIn("Checks:", output["explanation"])

    def test_tree_stop_output_for_unimplemented_leaf(self):
        agent = self._make_agent()
        agent.tree_result = TreeResult(
            method="trimmed_ipw",
            implemented=False,
            warnings=["trimmed_ipw is not implemented yet; this study cannot be estimated."],
            steps=[TreeStep("Leaf: trimmed IPW", "leaf", "selected (not implemented)")],
        )

        output = agent._tree_stop_output()

        self.assertIn("trimmed_ipw is not implemented yet", output["explanation"])

    @patch('cais.agent.run_decision_tree_v2')
    def test_run_analysis_stops_before_controls_and_execution(self, mock_tree):
        agent = self._make_agent()
        agent.analyse_dataset = MagicMock()
        agent.select_controls = MagicMock()
        agent.clean_dataset = MagicMock()
        agent.execute_method = MagicMock()
        mock_tree.return_value = TreeResult(ended=True, end_reason="Group composition is not stable; pipeline stopped.")

        output = agent.run_analysis("Does treatment affect outcome?", method_selection="tree_v2")

        self.assertEqual(output["results"]["status"], "not_estimated")
        agent.select_controls.assert_not_called()
        agent.clean_dataset.assert_not_called()
        agent.execute_method.assert_not_called()

    @patch('cais.agent.run_decision_tree_v2')
    def test_run_analysis_continues_for_implemented_leaf(self, mock_tree):
        agent = self._make_agent()
        agent.analyse_dataset = MagicMock()
        agent.select_controls = MagicMock()
        agent.clean_dataset = MagicMock()
        agent.execute_method = MagicMock(return_value={"results": {}, "explanation": "done"})
        mock_tree.return_value = TreeResult(method="difference_in_differences", implemented=True)

        output = agent.run_analysis("Does treatment affect outcome?", method_selection="tree_v2")

        agent.select_controls.assert_called_once()
        agent.clean_dataset.assert_called_once()
        agent.execute_method.assert_called_once()
        self.assertEqual(output["explanation"], "done")


if __name__ == '__main__':
    unittest.main()
