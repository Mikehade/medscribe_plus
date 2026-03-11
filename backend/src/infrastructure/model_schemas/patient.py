ExtractEHRFieldsGuidedJson = {
    "tools": [
        {
            "toolSpec": {
                "name": "extract_soap_fields",
                "description": (
                    "Extract the five standard SOAP note fields from a clinical note, "
                    "regardless of formatting. Return empty string for any field that is "
                    "genuinely absent. Do not fabricate content."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "subjective": {
                                "type": "string",
                                "description": "Chief complaint and subjective findings reported by the patient."
                            },
                            "objective": {
                                "type": "string",
                                "description": "Objective findings including vitals, lab results, and current medications."
                            },
                            "assessment": {
                                "type": "string",
                                "description": "Clinical assessment and/or diagnosis."
                            },
                            "plan": {
                                "type": "string",
                                "description": "Treatment plan including medication changes and clinical instructions."
                            },
                            "follow_up": {
                                "type": "string",
                                "description": "Follow-up timing or instructions."
                            }
                        },
                        "required": ["subjective", "objective", "assessment", "plan", "follow_up"]
                    }
                }
            }
        }
    ],
    "toolChoice": {
        "tool": {
            "name": "extract_soap_fields"
        }
    }
}