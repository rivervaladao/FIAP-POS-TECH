import os
import smtplib
import logging
from abc import ABC, abstractmethod
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()
# Configure basic logging if no handlers are configured
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class Alert(ABC):
    @abstractmethod
    def send_alert(self, message):
        pass

class ConsoleAlert(Alert):
    def send_alert(self, message):
        logging.debug(f"ALERT: {message}")

class EmailAlert(Alert):
    def __init__(self, recipient_email):
        self.recipient_email = recipient_email
        self.password = os.getenv("EMAIL_PASSWORD")
        if not self.password:
            logging.error("EMAIL_PASSWORD environment variable not set.")
            raise ValueError("EMAIL_PASSWORD environment variable not set.")
        
        self.from_email = os.getenv("EMAIL_FROM")
        if not self.from_email:
            logging.error("EMAIL_FROM environment variable not set.")
            raise ValueError("EMAIL_FROM environment variable not set.")

    def send_alert(self, message):
        msg = MIMEText(f"Alerta: objeto perigoso detectado ({message})!")
        msg["Subject"] = "Alerta de Segurança"
        msg["From"] = self.from_email
        msg["To"] = self.recipient_email

        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(self.from_email, self.password)
                server.sendmail(self.from_email, self.recipient_email, msg.as_string())
            logging.info(f"Email alert sent to {self.recipient_email} with message: {message}")
        except Exception as e:
            logging.error(f"Failed to send email alert to {self.recipient_email}: {e}")
