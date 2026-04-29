import smtplib
import socket
import json
import urllib.error
import urllib.request
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from html import escape
import logging

logger = logging.getLogger(__name__)
SMTP_HOST = "smtp.gmail.com"
SMTP_SSL_PORT = 465
SMTP_TIMEOUT_SECONDS = 15
RESEND_EMAILS_URL = "https://api.resend.com/emails"
RESEND_TIMEOUT_SECONDS = 15

def _extract_metric_value(metrics: dict, key: str):
    value = metrics.get(key, {})
    if isinstance(value, dict):
        return value.get("value", "N/A")
    return value if value is not None else "N/A"


def _build_executive_summary(best_model, roc_auc, accuracy, records) -> str:
    return (
        f"{best_model} is the current best model, with ROC-AUC {roc_auc} "
        f"and accuracy {accuracy}. The latest run processed {records} records "
        "to support faster prioritization of Dallas 311 requests."
    )


def _build_report_message(report_data: dict) -> tuple[str, str, str]:
    subject = "Dallas 311 ML Pipeline - Executive Report"
    metrics = report_data.get("metrics", {}) or {}
    best_model = report_data.get("best_model", "N/A")
    accuracy = _extract_metric_value(metrics, "accuracy")
    roc_auc = _extract_metric_value(metrics, "roc_auc")
    if roc_auc == "N/A":
        roc_auc = accuracy
        
    features = _extract_metric_value(metrics, "features")
    records = _extract_metric_value(metrics, "records")
    
    executive_summary = _build_executive_summary(best_model, roc_auc, accuracy, records)
    safe_best_model = escape(str(best_model))
    safe_accuracy = escape(str(accuracy))
    safe_roc_auc = escape(str(roc_auc))
    safe_features = escape(str(features))
    safe_records = escape(str(records))
    safe_summary = escape(str(executive_summary))

    text_content = (
        f"Dallas 311 Performance Summary\n"
        f"Best Algorithm: {best_model}\n"
        f"Accuracy: {accuracy}\n"
        f"ROC-AUC: {roc_auc}\n"
        f"Predictors Configured: {features}\n"
        f"Data Volume: {records}\n\n"
        f"Executive Summary:\n{executive_summary}\n\n"
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
                                <div style="font-size: 32px; font-weight: bold; color: #1e3a8a; margin-top: 4px;">{safe_best_model}</div>
                            </td>
                        </tr>
                    </table>

                    <!-- Two Column Grid -->
                    <table cellpadding="0" cellspacing="0" border="0" width="100%" style="margin-bottom: 24px;">
                        <tr>
                            <td width="48%" style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; text-align: center;">
                                <span style="font-size: 11px; font-weight: bold; text-transform: uppercase; color: #64748b;">ROC-AUC</span>
                                <div style="font-size: 24px; font-weight: bold; color: #0f172a; margin-top: 4px;">{safe_roc_auc}</div>
                            </td>
                            <td width="4%"></td>
                            <td width="48%" style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; text-align: center;">
                                <span style="font-size: 11px; font-weight: bold; text-transform: uppercase; color: #64748b;">Accuracy</span>
                                <div style="font-size: 24px; font-weight: bold; color: #0f172a; margin-top: 4px;">{safe_accuracy}</div>
                            </td>
                        </tr>
                    </table>

                    <!-- Additional Stats -->
                    <table cellpadding="0" cellspacing="0" border="0" width="100%" style="margin-top: 8px;">
                        <tr>
                            <td style="padding: 12px 16px; font-size: 14px; font-weight: bold; color: #475569; border-bottom: 1px solid #f1f5f9;">Predictors Configured</td>
                            <td style="padding: 12px 16px; font-size: 14px; font-weight: bold; color: #0f172a; text-align: right; border-bottom: 1px solid #f1f5f9;">{safe_features}</td>
                        </tr>
                        <tr>
                            <td style="padding: 12px 16px; font-size: 14px; font-weight: bold; color: #475569;">Data Volume</td>
                            <td style="padding: 12px 16px; font-size: 14px; font-weight: bold; color: #0f172a; text-align: right;">{safe_records}</td>
                        </tr>
                    </table>

                    <!-- Summary -->
                    <table cellpadding="0" cellspacing="0" border="0" width="100%" style="background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 12px; padding: 16px; margin-top: 24px;">
                        <tr>
                            <td>
                                <strong style="color: #0f172a; font-size: 14px; display: block; margin-bottom: 6px;">Executive Insight</strong>
                                <p style="font-size: 14px; color: #334155; margin: 0;">{safe_summary}</p>
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
    return subject, text_content, html_content


def send_resend_report(api_key: str, sender_email: str, recipient_email: str, report_data: dict) -> tuple[bool, str]:
    """
    Dispatches the ML metrics summary through Resend's HTTPS email API.
    """
    logger.info("Preparing to send dashboard summary through Resend to %s...", recipient_email)

    subject, text_content, html_content = _build_report_message(report_data)
    payload = {
        "from": sender_email,
        "to": [recipient_email] if isinstance(recipient_email, str) else list(recipient_email),
        "subject": subject,
        "html": html_content,
        "text": text_content,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        RESEND_EMAILS_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Dallas311Dashboard/1.0",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=RESEND_TIMEOUT_SECONDS) as response:
            response_body = response.read().decode("utf-8")
        logger.info("Report successfully sent through Resend: %s", response_body)
        return True, ""
    except urllib.error.HTTPError as e:
        response_body = e.read().decode("utf-8", errors="replace")
        error_message = f"Resend API returned HTTP {e.code}: {response_body}"
        logger.error("Resend dispatch failure: %s", error_message)
        return False, error_message
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        error_message = f"Could not reach Resend API over HTTPS: {e}"
        logger.error("Resend dispatch failure: %s", error_message)
        return False, error_message


def send_gmail_report(sender_email: str, app_password: str, recipient_email: str, report_data: dict) -> tuple[bool, str]:
    """
    Connects to Gmail's SMTP server using an App Password and dispatches the ML metrics summary.
    """
    logger.info(f"Preparing to send dashboard summary from {sender_email} to {recipient_email}...")

    subject, text_content, html_content = _build_report_message(report_data)

    msg = MIMEMultipart("alternative")
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg["Subject"] = subject
    msg.attach(MIMEText(text_content, "plain"))
    msg.attach(MIMEText(html_content, "html"))

    recipients = [recipient_email] if isinstance(recipient_email, str) else list(recipient_email)

    try:
        logger.info("Opening SSL link to %s on port %s...", SMTP_HOST, SMTP_SSL_PORT)
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_SSL_PORT, timeout=SMTP_TIMEOUT_SECONDS) as server:
            server.ehlo()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, recipients, msg.as_string())
        logger.info("Report successfully sent over secure protocol!")
        return True, ""
    except smtplib.SMTPAuthenticationError:
        error_message = "Gmail rejected the login. Use the Gmail address and a 16-character App Password."
        logger.error("SMTP Dispatch failure: %s", error_message)
        return False, error_message
    except (socket.timeout, TimeoutError):
        error_message = (
            "Timed out connecting to Gmail SMTP. If this is running on Render free tier, "
            "outbound SMTP ports 25, 465, and 587 are blocked; use a paid instance or an email HTTP API."
        )
        logger.error("SMTP Dispatch failure: %s", error_message)
        return False, error_message
    except Exception as e:
        error_message = str(e)
        logger.error(f"SMTP Dispatch failure: {error_message}")
        return False, error_message
