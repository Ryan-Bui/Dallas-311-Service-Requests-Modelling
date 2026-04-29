import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

logger = logging.getLogger(__name__)

def _extract_metric_value(metrics: dict, key: str):
    value = metrics.get(key, {})
    if isinstance(value, dict):
        return value.get("value", "N/A")
    return value if value is not None else "N/A"


def send_gmail_report(sender_email: str, app_password: str, recipient_email: str, report_data: dict) -> tuple[bool, str]:
    """
    Connects to Gmail's SMTP server using an App Password and dispatches the ML metrics summary.
    """
    logger.info(f"Preparing to send dashboard summary from {sender_email} to {recipient_email}...")
    
    # 1. Construct HTML Email Template
    subject = "Dallas 311 ML Pipeline - Executive Report"
    metrics = report_data.get("metrics", {}) or {}
    best_model = report_data.get("best_model", "N/A")
    accuracy = _extract_metric_value(metrics, "accuracy")
    roc_auc = _extract_metric_value(metrics, "roc_auc")
    if roc_auc == "N/A":
        roc_auc = accuracy
        
    features = _extract_metric_value(metrics, "features")
    records = _extract_metric_value(metrics, "records")
    
    # 1b. Generate LLM Executive Summary
    llm_summary = "Automated predictive analysis completed successfully."
    try:
        from inference.llm_factory import get_llm
        llm = get_llm(temperature=0.7)
        prompt = f"""You are a senior data scientist analyzing the Dallas 311 ML pipeline performance. 
        Write a hyper-concise executive summary (maximum 2 sentences) describing what this means for operational efficiency.
        Do NOT use consultant jargon, fluff, or complex terminology. Speak plainly.
        
        Metrics:
        - Best Model: {best_model}
        - ROC-AUC: {roc_auc}
        - Accuracy: {accuracy}
        - Data Volume: {records} records processed.
        """
        response = llm.invoke(prompt)
        llm_summary = response.content.strip()
    except Exception as e:
        logger.warning(f"Failed to generate LLM summary: {e}")

    text_content = (
        f"Dallas 311 Performance Summary\n"
        f"Best Algorithm: {best_model}\n"
        f"Accuracy: {accuracy}\n"
        f"ROC-AUC: {roc_auc}\n"
        f"Predictors Configured: {features}\n"
        f"Data Volume: {records}\n\n"
        f"Executive Summary:\n{llm_summary}\n\n"
        f"This automated email was requested from the secure administrator portal of the Dallas 311 Service Requests Platform."
    )

    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color: #f8fafc; color: #1e293b; padding: 40px 20px; line-height: 1.5;">
        <table cellpadding="0" cellspacing="0" border="0" width="100%" style="max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 16px; overflow: hidden; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);">
            <!-- Header -->
            <tr>
                <td style="background: #1e3a8a; padding: 32px 24px; text-align: center;">
                    <h2 style="color: #ffffff; margin: 0; font-size: 24px; font-weight: bold;">Dallas 311 Performance Summary</h2>
                    <p style="color: #93c5fd; margin: 8px 0 0 0; font-size: 14px;">Predictive Analytics Intelligence Suite</p>
                </td>
            </tr>
            
            <!-- Content -->
            <tr>
                <td style="padding: 32px 24px;">
                    <!-- Hero Metric -->
                    <table cellpadding="0" cellspacing="0" border="0" width="100%" style="background: #eff6ff; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 24px; border: 1px solid #dbeafe;">
                        <tr>
                            <td>
                                <span style="font-size: 13px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.05em; color: #1d4ed8;">Best Algorithm</span>
                                <div style="font-size: 32px; font-weight: bold; color: #1e3a8a; margin-top: 4px;">{best_model}</div>
                            </td>
                        </tr>
                    </table>

                    <!-- Two Column Grid -->
                    <table cellpadding="0" cellspacing="0" border="0" width="100%" style="margin-bottom: 24px;">
                        <tr>
                            <td width="48%" style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; text-align: center;">
                                <span style="font-size: 11px; font-weight: bold; text-transform: uppercase; color: #64748b;">ROC-AUC</span>
                                <div style="font-size: 24px; font-weight: bold; color: #0f172a; margin-top: 4px;">{roc_auc}</div>
                            </td>
                            <td width="4%"></td>
                            <td width="48%" style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; text-align: center;">
                                <span style="font-size: 11px; font-weight: bold; text-transform: uppercase; color: #64748b;">Accuracy</span>
                                <div style="font-size: 24px; font-weight: bold; color: #0f172a; margin-top: 4px;">{accuracy}</div>
                            </td>
                        </tr>
                    </table>

                    <!-- Additional Stats -->
                    <table cellpadding="0" cellspacing="0" border="0" width="100%" style="margin-top: 8px;">
                        <tr>
                            <td style="padding: 12px 16px; font-size: 14px; font-weight: bold; color: #475569; border-bottom: 1px solid #f1f5f9;">Predictors Configured</td>
                            <td style="padding: 12px 16px; font-size: 14px; font-weight: bold; color: #0f172a; text-align: right; border-bottom: 1px solid #f1f5f9;">{features}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px 16px; font-size: 14px; font-weight: bold; color: #475569;">Data Volume</td>
                            <td style="padding: 12px 16px; font-size: 14px; font-weight: bold; color: #0f172a; text-align: right;">{records}</td>
                        </tr>
                    </table>

                    <!-- AI Summary -->
                    <table cellpadding="0" cellspacing="0" border="0" width="100%" style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 12px; padding: 16px; margin-top: 24px;">
                        <tr>
                            <td>
                                <strong style="color: #0f172a; font-size: 14px; display: block; margin-bottom: 6px;">🧠 AI Executive Insight</strong>
                                <p style="font-size: 14px; color: #334155; margin: 0;">{llm_summary}</p>
                            </td>
                        </tr>
                    </table>

                    <div style="margin-top: 32px; padding-top: 24px; border-top: 1px solid #e2e8f0; text-align: center;">
                        <p style="font-size: 12px; color: #94a3b8; margin: 0;">
                            This automated email was requested from the secure administrator portal of the Dallas 311 Service Requests Platform.
                        </p>
                    </div>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(text_content, "plain"))
    msg.attach(MIMEText(html_content, "html"))

    recipients = [recipient_email] if isinstance(recipient_email, str) else list(recipient_email)

    try:
        logger.info("Opening SSL link to smtp.gmail.com on port 465...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.ehlo()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, recipients, msg.as_string())
        logger.info("Report successfully sent over secure protocol!")
        return True, ""
    except Exception as e:
        error_message = str(e)
        logger.error(f"SMTP Dispatch failure: {error_message}")
        return False, error_message
