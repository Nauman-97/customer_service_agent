import sqlite3
import time
from pathlib import Path


from utils import get_logger

history_logger = get_logger(name= "email_history", filename= "email_history.log", level= "debug")

class EmailHistory:
    session_id = 0
    message = ""
    # ---------------------------------------------------------
    # INIT
    # ---------------------------------------------------------
    def __init__(self, max_retries:int, retry_delay: int, sid_prefix:str = ""):
        self.DB_DIR = Path("data/email_history_database") 
        self.DB_NAME = "email_history.db"
        self.DB_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = self.DB_DIR / self.DB_NAME
        self.MAX_RETRIES = max_retries
        self.RETRY_DELAY = retry_delay
        self.sid_prefix = sid_prefix

        self._init_db()

    # ---------------------------------------------------------
    # INIT DATABASE
    # ---------------------------------------------------------
    def _init_db(self):

        def operation(conn):

            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sid Text,
                    email_address TEXT,
                    subject TEXT,
                    body TEXT,
                    summary TEXT,
                    reply TEXT,
                    category TEXT,
                    priority TEXT,
                    spam_probability REAL,
                    escalate INTEGER,
                    close INTEGER
                )
            """)

        self._execute_with_retry(operation, instance= "create_table")
    def _session_id(self):
        return self.sid_prefix +"-"+ str(self.session_id) 
    # ---------------------------------------------------------
    # CONNECTION
    # ---------------------------------------------------------
    def _connect(self):
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30
        )

        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ---------------------------------------------------------
    # RETRY WRAPPER (PROPER TRANSACTION HANDLING)
    # ---------------------------------------------------------
    def _execute_with_retry(self, operation, instance:str):
        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            conn = None
            try:
                conn = self._connect()
                result = operation(conn)
                conn.commit()
                history_logger.debug(f"Successful for {instance} in {attempt} attempt/s")
                return result
            except sqlite3.OperationalError as e:
                last_error = e
                if conn:
                    conn.rollback()
                history_logger.error(f"[SQLITE] Failed for {instance} Attempt {attempt}/{self.MAX_RETRIES}: \n{e}\n")
                time.sleep(self.RETRY_DELAY)
            except Exception as e:
                last_error = e
                if conn:
                    conn.rollback()
                history_logger.error(f"[GENERAL] Failed for {instance} Attempt {attempt}/{self.MAX_RETRIES}: \n{e}\n")
                time.sleep(self.RETRY_DELAY)
            finally:
                if conn:
                    conn.close()
        history_logger.critical(f"FAILED for <{instance}>  After {self.MAX_RETRIES} retries.  \n{last_error}\n")
        raise TimeoutError(f"FAILED for <{instance}>  After {self.MAX_RETRIES} retries.  \n{last_error}\n")

    # ---------------------------------------------------------
    # INSERT HISTORY
    # ---------------------------------------------------------

    def add_history(self, history):
        def operation(conn):
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO history (sid, email_address, subject, body, 
                                     summary, reply, category, priority, 
                                     spam_probability, escalate, close)
                VALUES (?, ?, ?, ?, 
                        ?, ?, ?, ?, 
                        ?, ?, ?)
            """, (self._session_id(), history.get("email_address"), history.get("subject"), history.get("body"),
                  
                  history.get("summary"), history.get("reply"), history.get("category"), history.get("priority"),

                  history.get("spam_probability"), int(bool(history.get("escalate"))), int(bool(history.get("close")))))

            return True
        instance= f"SID {self._session_id()} func: add_history"
        return self._execute_with_retry(operation=operation,instance=instance)


    # ---------------------------------------------------------
    # GET HISTORY 
    # ---------------------------------------------------------
    def get_history(self, version: str = "", email_address: str= None, limit: int = 3)-> dict:
        self.message = ""
        def operation(conn):

            cursor = conn.cursor()
            if version == "classifier":
                self.session_id += 1
            # ---------------------------------------------------------
            # CLASSIFIER VERSION
            # ---------------------------------------------------------
            if email_address:
                if version == "classifier":
                    cursor.execute("""
                        SELECT subject, body, reply, spam_probability
                        FROM (
                            SELECT id, subject, body, reply, spam_probability
                            FROM history
                            WHERE email_address = ?
                            ORDER BY id DESC
                            LIMIT ?)
                        ORDER BY id ASC
                    """, (str(email_address), limit))

                    rows = cursor.fetchall()

                    if not rows:
                        return {}
                        
                    output = {}
                    for i, r in enumerate(rows, 1):
                        entry = {
                            "subject": r[0],
                            "body": r[1],
                            "reply": r[2],
                            "spam_probability": r[3]
                        }
                        output = output | {f"{i}": entry}
                    return output
            # ---------------------------------------------------------
            # RESPONDER VERSION
            # ---------------------------------------------------------
                elif version == "responder":
                    cursor.execute(
                    """SELECT subject, body, reply, category, priority
                        FROM (
                            SELECT id, subject, body, reply, category, priority
                            FROM history
                            WHERE email_address = ?
                            ORDER BY id DESC
                            LIMIT ?)
                        ORDER BY id ASC""", (str(email_address), limit))
                    rows = cursor.fetchall()
                    if not rows:
                        return {}
                    output = {}
                    for i, r in enumerate(rows, 1):
                        entry = {
                            "subject": r[0],
                            "body": r[1],
                            "reply": r[2],
                            "category": r[3],
                            "priority": r[4]
                        }
                        output = output | {f"{i}": entry}
                    return output

            # ---------------------------------------------------------
            # INVALID VERSION
            # ---------------------------------------------------------
                else:
                    raise ValueError(f"SID: {self._session_id()} Invalid version. Use 'classifier' or 'responder'. Or give missing parameters ")
        instance = f"SID: {self._session_id()} func: get_history for {version}"
        return self._execute_with_retry(operation, instance= instance)