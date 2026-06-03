import unittest
from unittest.mock import MagicMock, patch
import os
import json
from clinical_agent.tools import check_drug_interactions
from clinical_agent.parser import LlamaCloudParser
from clinical_agent.nodes import parse_json_safely
from clinical_agent.graph import route_based_on_action

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

        # Test no interactions for clean lists
        meds_clean = ["TAB. RACIPER 40MG", "TAB. EMESET 4MG"]
        alerts_clean = check_drug_interactions(meds_clean)
        self.assertEqual(len(alerts_clean), 0)

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
        state = {
            "is_finished": True,
            "iteration_count": 2,
            "extracted_data": {}
        }
        route = route_based_on_action(state)
        self.assertEqual(route, "synthesizer")

        # Mock State plan action READ_DOCUMENTS
        state2 = {
            "is_finished": False,
            "iteration_count": 2,
            "extracted_data": {"_next_action": "READ_DOCUMENTS"}
        }
        route2 = route_based_on_action(state2)
        self.assertEqual(route2, "reader")

        # Mock State plan action VERIFY_SAFETY
        state3 = {
            "is_finished": False,
            "iteration_count": 4,
            "extracted_data": {"_next_action": "VERIFY_SAFETY"}
        }
        route3 = route_based_on_action(state3)
        self.assertEqual(route3, "verifier")

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

if __name__ == "__main__":
    unittest.main()
