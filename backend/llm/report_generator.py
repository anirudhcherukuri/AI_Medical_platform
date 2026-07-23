import os
from typing import Dict, Any

try:
    from google import genai as genai_new
    from google.genai import types
    GENAI_NEW_API = True
except ImportError:
    genai_new = None
    GENAI_NEW_API = False

from backend.config import GEMINI_API_KEY, PREFERRED_LLM_MODEL


class LLMMedicalReportGenerator:
    """
    LLM-powered Medical Radiology Report Generator.
    Integrates with Google Gemini API (google-genai SDK) with fallback to structured clinical logic.
    """
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.client_available = False
        self.client = None

        if self.api_key and GENAI_NEW_API:
            try:
                self.client = genai_new.Client(api_key=self.api_key)
                self.client_available = True
                print("[LLMReportGenerator] Initialized Google Gemini (google-genai) Client successfully.")
            except Exception as e:
                print(f"[LLMReportGenerator] Failed to configure Gemini API ({e}). Falling back to Clinical Logic Engine.")
        else:
            print("[LLMReportGenerator] Operating in Fallback Standalone Mode (Clinical Logic Engine).")

    def generate_report(
        self,
        predicted_class: str,
        confidence: float,
        probabilities: Dict[str, float],
        spatial_metrics: Dict[str, Any],
        patient_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if patient_info is None:
            patient_info = {
                "patient_id": "P-MED-9921",
                "age": 45,
                "gender": "Unspecified",
                "scan_type": "Chest X-Ray PA View"
            }

        prompt = self._construct_prompt(predicted_class, confidence, probabilities, spatial_metrics, patient_info)

        if self.client_available:
            try:
                response = self.client.models.generate_content(
                    model=PREFERRED_LLM_MODEL,
                    contents=prompt
                )
                generated_text = response.text
                return self._parse_llm_response(generated_text, predicted_class, confidence, spatial_metrics)
            except Exception as e:
                print(f"[LLMReportGenerator] Gemini API call error: {e}. Executing rule fallback.")

        return self._generate_rule_based_report(predicted_class, confidence, probabilities, spatial_metrics, patient_info)

    def _construct_prompt(self, predicted_class, confidence, probabilities, spatial_metrics, patient_info):
        return f"""
You are an expert AI Radiologist Assistant generating a formal structured diagnostic report.
Analyze the following patient parameters and Deep Learning / Explainable AI outputs:

PATIENT METADATA:
- Patient ID: {patient_info.get('patient_id')}
- Age: {patient_info.get('age')} | Gender: {patient_info.get('gender')}
- Modality: {patient_info.get('scan_type')}

DEEP LEARNING MODEL DIAGNOSIS:
- Primary Finding: {predicted_class}
- Model Confidence Score: {confidence * 100:.2f}%
- Class Probabilities: Normal={probabilities.get('Normal', 0)*100:.1f}%, Pneumonia={probabilities.get('Pneumonia', 0)*100:.1f}%, COVID-19={probabilities.get('COVID-19', 0)*100:.1f}%

EXPLAINABLE AI (Grad-CAM) SPATIAL ATTRIBUTION:
- Peak Intensity Location: {spatial_metrics.get('peak_location_normalized')}
- Affected Lung Region Coverage: {spatial_metrics.get('activation_coverage_pct')}%
- Spatial Severity Metric: {spatial_metrics.get('severity_score')}

INSTRUCTIONS:
Generate a professional, structured clinical report including:
1. CLINICAL SUMMARY
2. RADIOLOGICAL FINDINGS (pulmonary opacities, focal consolidations, or clear fields)
3. EXPLAINABLE AI INTERPRETATION (linking Grad-CAM heatmap regions to anatomical pathology)
4. IMPRESSION & RECOMMENDATION (suggested follow-up: CT scans, PCR, bloodwork, or supportive care)

Be precise, objective, and maintain high standards of clinical communication.
"""

    def _parse_llm_response(self, text, predicted_class, confidence, spatial_metrics):
        risk_level = "Low Risk" if predicted_class == "Normal" else ("Moderate Risk" if confidence < 0.85 else "High Risk")
        return {
            "source": "Gemini 1.5 Flash AI",
            "summary": f"Primary Diagnostic Impression: {predicted_class} (Confidence: {confidence*100:.1f}%).",
            "full_report_text": text,
            "risk_level": risk_level,
            "disclaimer": "AI-generated report for clinical decision support. Requires verification by a board-certified Radiologist."
        }

    def _generate_rule_based_report(self, predicted_class, confidence, probabilities, spatial_metrics, patient_info):
        coverage = spatial_metrics.get("activation_coverage_pct", 0)

        if predicted_class == "Normal":
            risk = "Low Risk"
            findings = "Both lung fields are clear without focal pulmonary consolidation, interstitial thickening, or pleural effusion. Cardiomediastinal silhouette is within normal limits."
            xai_text = f"Grad-CAM feature attribution shows diffuse uniform background response (coverage {coverage}%) without localized pathologic hotspots."
            recommendation = "No active pulmonary disease detected. Routine clinical follow-up as indicated."
        elif predicted_class == "Pneumonia":
            risk = "High Risk" if confidence > 0.8 else "Moderate Risk"
            findings = f"Focal pulmonary opacity and patchy airspace consolidation detected. Model probability: {confidence*100:.1f}% for bacterial/viral pneumonia."
            xai_text = f"Grad-CAM neural heatmap highlights focal opacity region occupying approximately {coverage}% of lung field at coordinates {spatial_metrics.get('peak_location_normalized')}."
            recommendation = "Recommend clinical correlation, sputum culture, CBC, and follow-up lateral X-Ray or chest CT as clinically indicated."
        else:
            risk = "High Risk"
            findings = f"Bilateral ground-glass opacities and peripheral lung involvement characteristic of COVID-19 pulmonary manifestation. Probability: {confidence*100:.1f}%."
            xai_text = f"Grad-CAM visual explanation identifies characteristic peripheral subpleural activation zones covering {coverage}% of spatial lung area."
            recommendation = "Immediate isolation protocols, RT-PCR testing, arterial blood gas analysis, and respiratory monitoring recommended."

        full_text = f"""================================================================================
RADIOLOGY EVALUATION REPORT (AI-ASSISTED)
================================================================================
PATIENT ID: {patient_info.get('patient_id')} | AGE: {patient_info.get('age')} | MODALITY: {patient_info.get('scan_type')}
--------------------------------------------------------------------------------
1. CLINICAL IMPRESSION:
   Primary Finding: {predicted_class.upper()}
   AI Diagnostic Confidence: {confidence*100:.2f}%
   Risk Stratification: {risk}

2. RADIOLOGICAL FINDINGS:
   {findings}

3. EXPLAINABLE AI (Grad-CAM) INTERPRETATION:
   {xai_text}

4. RECOMMENDED CLINICAL PLAN:
   {recommendation}
================================================================================
DISCLAIMER: This diagnostic report is synthesized by the AI Medical Intelligence Platform
for preliminary screening and decision support. Final diagnostic verification must be
conducted by a licensed medical practitioner.
"""
        return {
            "source": "AI Clinical Logic Engine (Fallback)",
            "summary": f"Primary Diagnostic Impression: {predicted_class} ({confidence*100:.1f}% confidence).",
            "full_report_text": full_text,
            "risk_level": risk,
            "disclaimer": "AI-generated report for clinical decision support. Requires verification by a board-certified Radiologist."
        }


_report_generator = None

def get_report_generator() -> LLMMedicalReportGenerator:
    global _report_generator
    if _report_generator is None:
        _report_generator = LLMMedicalReportGenerator()
    return _report_generator
