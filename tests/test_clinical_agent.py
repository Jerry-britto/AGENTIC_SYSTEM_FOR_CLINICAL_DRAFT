import unittest
from unittest.mock import MagicMock, patch
import os
import json
from clinical_agent.tools import check_drug_interactions
from clinical_agent.parser import LlamaCloudParser
from clinical_agent.nodes import parse_json_safely, verify_no_hallucinations, planner_node
from clinical_agent.graph import route_based_on_action
from clinical_agent.state import AgentState

class TestClinicalAgent(unittest.TestCase):
    
    def test_drug_interaction_lookup(self):
        """
        Test that mock drug-drug interactions are successfully caught by our tools module.
        """
        # Test Warfarin and Aspirin interaction
        meds = ["TAB. ASPIRIN 75MG", "TAB. WARFARIN 5MG"]
        alerts = check_drug_interactions(meds)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["severity"], "CRITICAL")
        self.assertIn("hemorrhage", alerts[0]["mechanism"].lower())

        # Test Lisinopril and Potassium interaction
        meds2 = ["TAB. LISINOPRIL 10MG", "POTASSIUM SUPPLEMENTS 10MEQ"]
        alerts2 = check_drug_interactions(meds2)
        self.assertEqual(len(alerts2), 1)
        self.assertEqual(alerts2[0]["severity"], "CRITICAL")
        self.assertIn("hyperkalemia", alerts2[0]["mechanism"].lower())

        # Test Loperamide and Oflox TZ clinical safety warning
        meds3 = ["TAB. LOPIRAMIDE 2MG", "TAB. OFLOX TZ"]
        alerts3 = check_drug_interactions(meds3)
        self.assertEqual(len(alerts3), 1)
        self.assertEqual(alerts3[0]["severity"], "HIGH WARNING")
        self.assertIn("megacolon", alerts3[0]["mechanism"].lower())

    def test_json_safe_parsing(self):
        """
        Test that JSON string parsing safely removes markdown wrappers.
        """
        raw_json_str = """```json
        {
            "name": "Test Patient",
            "val": 123
        }
        ```"""
        parsed = parse_json_safely(raw_json_str)
        self.assertEqual(parsed["name"], "Test Patient")
        self.assertEqual(parsed["val"], 123)

    def test_graph_routing(self):
        """
        Test state graph routes correctly based on planner's choices.
        """
        # Mock State finished
        state: AgentState = {
            "is_finished": True,
            "iteration_count": 2,
            "extracted_data": {},
            "parsed_pages": [],
            "medication_changes": [],
            "safety_warnings": [],
            "clinician_review_flags": [],
            "current_plan": "",
            "trace_steps": [],
            "completed_phases": [],
            "retrieved_pages": [],
            "hallucination_report": {},
            "output_markdown": "",
            "output_json": {}
        }
        route = route_based_on_action(state)
        self.assertEqual(route, "synthesizer")

        # Mock State plan action READ_DOCUMENTS
        state2 = state.copy()
        state2["is_finished"] = False
        state2["extracted_data"] = {"_next_action": "READ_DOCUMENTS"}
        route2 = route_based_on_action(state2)
        self.assertEqual(route2, "reader")

    @patch("os.path.exists")
    @patch("builtins.open")
    def test_parser_uses_cache(self, mock_open, mock_exists):
        """
        Test that LlamaCloudParser correctly prioritizes local cache if it is found.
        """
        mock_exists.return_value = True
        mock_open_instance = mock_open.return_value.__enter__.return_value
        mock_open_instance.read.return_value = json.dumps({
            "num_pages": 1,
            "pages": [{"page_number": 1, "markdown": "# Test Note"}]
        })
        
        parser = LlamaCloudParser()
        data = parser.parse_pdf("data/patient_records.pdf")
        
        self.assertEqual(data["num_pages"], 1)
        self.assertEqual(data["pages"][0]["markdown"], "# Test Note")

    def test_page_classification(self):
        """
        Test that LlamaCloudParser correctly identifies and filters reference guidelines page,
        but retains patient-specific pages.
        """
        parser = LlamaCloudParser()
        pages = [
            {
                "page_number": 1,
                "markdown": "CLINICAL RECORD - David Miller\nMRN: 99827\nAdmission Date: 2026-05-01\nDiagnoses: Gastroenteritis"
            },
            {
                "page_number": 2,
                "markdown": "Clinical Guideline and Dosing Table for Antibiotics\nStandard hospital policy and reference protocols for general adult patients."
            }
        ]
        filtered = parser.classify_and_filter_pages(pages)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["page_number"], 1)

    def test_loop_prevention(self):
        """
        Test that planner_node overrides repeated actions to prevent infinite loops.
        """
        # Mock planner state attempting to run reconcile for a second time
        state: AgentState = {
            "parsed_pages": [],
            "extracted_data": {},
            "medication_changes": [],
            "safety_warnings": [],
            "clinician_review_flags": [],
            "current_plan": "Checking meds",
            "trace_steps": [],
            "iteration_count": 2,
            "completed_phases": ["reconcile"],
            "retrieved_pages": [],
            "hallucination_report": {},
            "output_markdown": "",
            "output_json": {},
            "is_finished": False
        }
        
        # We patch call_groq_llm to return the action RECONCILE_MEDICATIONS
        with patch("clinical_agent.nodes.call_groq_llm") as mock_call:
            mock_call.return_value = json.dumps({
                "reasoning": "Reconcile meds again.",
                "action": "RECONCILE_MEDICATIONS",
                "action_parameters": {}
            })
            
            result_state = planner_node(state)
            # Should override RECONCILE_MEDICATIONS to VERIFY_SAFETY
            self.assertEqual(result_state["extracted_data"]["_next_action"], "VERIFY_SAFETY")

    @patch("clinical_agent.nodes.call_groq_llm")
    def test_hallucination_guardrail(self, mock_llm):
        """
        Test that verify_no_hallucinations detects drugs not mentioned in patient pages.
        """
        # Page has Lasix but not FictionalDrug
        pages = [{"page_number": 1, "markdown": "Patient was given Lasix."}]
        state: AgentState = {
            "parsed_pages": pages,
            "extracted_data": {
                "discharge_medications": ["Lasix", "FictionalDrug"]
            },
            "medication_changes": [],
            "safety_warnings": [],
            "clinician_review_flags": [],
            "current_plan": "",
            "trace_steps": [],
            "iteration_count": 1,
            "completed_phases": [],
            "retrieved_pages": [],
            "hallucination_report": {},
            "output_markdown": "",
            "output_json": {},
            "is_finished": False
        }
        
        # Mock LLM to verify that FictionalDrug is NOT in document
        mock_llm.return_value = "NO"
        
        result_state = verify_no_hallucinations(state)
        report = result_state["hallucination_report"]
        
        self.assertIn("FictionalDrug", report["hallucinations_detected"])
        self.assertIn("Lasix", report["checked_items"])
        self.assertEqual(report["checked_items"]["Lasix"], "VERIFIED (Direct Substring)")
        
        # Verify a warning was added
        self.assertTrue(any("FictionalDrug" in w["message"] for w in result_state["safety_warnings"]))

if __name__ == "__main__":
    unittest.main()
