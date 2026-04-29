import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

logger = logging.getLogger(__name__)

def send_gmail_report(sender_email: str, app_password: str, recipient_email: str, report_data: dict) -> bool:
    """
    Connects to Gmail's SMTP server using an App Password and dispatches the ML metrics summary.
    """
    logger.info(f"Preparing to send dashboard summary from {sender_email} to {recipient_email}...")
    
    # 1. Construct HTML Email Template
    subject = f"Dallas 311 ML Pipeline - Executive Report"
    
    metrics = report_data.get("metrics", {})
    best_model = report_data.get("best_model", "N/A")
    accuracy = metrics.get("accuracy", {}).get("value", "N/A")
    roc_auc = metrics.get("accuracy", {}).get("value", "N/A") # Placeholder if AUC distinct
    features = metrics.get("features", {}).get("value", "N/A")
    records = metrics.get("records", {}).get("value", "N/A")
    
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color: #f4f6f9; color: #333333; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background: #ffffff; padding: 20px; border-radius: 8px; border: 1px solid #e1e4e8;">
            <h2 style="color: #1a365d; border-bottom: 2px solid #3182ce; padding-bottom: 10px;">Dallas 311 Performance Summary</h2>
            <p>Here are the latest metrics from the predictive analytics intelligence suite:</p>
            
            <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                <tr style="background-color: #f7fafc;">
                    <td style="padding: 10px; font-weight: bold; border: 1px solid #e2e8f0;">Best Algorithm</td>
                    <td style="padding: 10px; border: 1px solid #e2e8f0;">{best_model}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; font-weight: bold; border: 1px solid #e2e8f0;">Accuracy (ROC-AUC)</td>
                    <td style="padding: 10px; border: 1px solid #e2e8f0;">{accuracy}</td>
                </tr>
                <tr style="background-color: #f7fafc;">
                    <td style="padding: 10px; font-weight: bold; border: 1px solid #e2e8f0;">Predictors Configured</td>
                    <td style="padding: 10px; border: 1px solid #e2e8f0;">{features}</td>
                </tr>
                <tr>
                    <td style="padding: 10px; font-weight: bold; border: 1px solid #e2e8f0;">Data Volume</td>
                    <td style="padding: 10px; border: 1px solid #e2e8f0;">{records}</td>
                </tr>
            </table>
            
            <p style="font-size: 12px; color: #718096; text-align: center; margin-top: 30px;">
                This automated email was requested from the secure administrator portal of the Dallas 311 Service Requests Platform.
            </p>
        </div>
    </body>
    </html>
    """

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(html_content, 'html'))

    # 2. Open Secure Connection and Dispatch
    try:
        logger.info("Opening SSL link to smtp.gmail.com on port 465...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        logger.info("Report successfully sent over secure protocol!")
        return True
    except Exception as e:
        logger.error(f"SMTP Dispatch failure: {e}")
        return False
