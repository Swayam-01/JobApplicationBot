import os
import json
import time
import logging
import numpy as np
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import spacy
import tensorflow as tf
from sentence_transformers import SentenceTransformer

# ---------------------------
# Environment & Logging Setup
# ---------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_AFFINITY'] = 'noverbose'
os.environ['OMP_NUM_THREADS'] = '1'
tf.get_logger().setLevel('ERROR')

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "..", "config", "config.json")
dotenv_path = os.path.join(script_dir, "..", "config", ".env")
resume_path = os.path.join(script_dir, "..", "data", "resume.txt")
log_path = os.path.join(script_dir, "..", "logs", "applications.log")

load_dotenv(dotenv_path)
with open(config_path, 'r') as f:
    config = json.load(f)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# ---------------------------
# Initialize NLP and AI Models
# ---------------------------
nlp = spacy.load("en_core_web_sm")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

with open(resume_path, 'r') as f:
    resume_text = f.read()

# ---------------------------
# Helper Functions
# ---------------------------
def parse_resume():
    """
    Analyze the resume using spaCy and extract skill entities.
    If no "SKILL" entities are found, fallback to the entire resume text.
    """
    doc = nlp(resume_text)
    skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    if not skills:
        return resume_text
    return ' '.join(skills)

def job_similarity(job_desc):
    """
    Calculate semantic similarity between the resume and job description.
    Returns a float similarity score.
    """
    resume_embedding = sentence_model.encode(parse_resume())
    job_embedding = sentence_model.encode(job_desc)
    return np.dot(resume_embedding, job_embedding)

def job_has_easy_apply(job_element):
    """
    Check if the job element's text contains "Easy Apply".
    Returns True if found, else False.
    """
    try:
        text = job_element.text
        return "Easy Apply" in text
    except Exception:
        return False

def extract_job_description(job_element):
    """
    Attempt to extract the job description text using multiple selectors.
    Returns a string (or empty if not found).
    """
    try:
        desc_elem = job_element.find_element(By.CSS_SELECTOR, "div.job-card-container__metadata")
        return desc_elem.text.strip()
    except Exception:
        try:
            desc_elem = job_element.find_element(By.XPATH, ".//div[contains(@class, 'job-card-list__insight')]")
            return desc_elem.text.strip()
        except Exception as e:
            logging.error(f"Could not extract job description: {str(e)}")
            return ""

# ---------------------------
# Selenium Web Automation Functions
# ---------------------------
def initialize_driver():
    """
    Initialize ChromeDriver with anti-detection and performance options.
    """
    try:
        service = Service(
            ChromeDriverManager().install(),
            service_args=['--verbose', '--log-path=chrome_debug.log']
        )
    except Exception as e:
        logging.warning(f"Auto-install failed: {str(e)}. Using local chromedriver.")
        service = Service(
            executable_path=os.path.join(script_dir, "..", "chromedriver.exe"),
            service_args=['--verbose', '--log-path=chrome_debug.log']
        )
    
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-webgl")
    options.add_argument("--disable-3d-apis")
    options.add_argument("--use-angle=swiftshader")
    options.add_argument("--disable-device-discovery-notifications")
    options.add_argument("--use-fake-device-for-media-stream")
    options.add_argument("--use-fake-ui-for-media-stream")
    options.add_argument("--disable-features=VizDisplayCompositor")
    options.add_argument("--enable-features=NetworkServiceInProcess")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--start-maximized")
    options.add_argument("--log-level=3")
    options.add_argument("--disable-logging")
    
    return webdriver.Chrome(service=service, options=options)

def linkedin_login(driver):
    """
    Log into LinkedIn using credentials from the .env file.
    Pauses for manual email verification if required.
    """
    try:
        print("Navigating to LinkedIn login page...")
        driver.get("https://www.linkedin.com/login")
        
        email = os.getenv("LINKEDIN_EMAIL")
        password = os.getenv("LINKEDIN_PASSWORD")
        if not email or not password:
            raise ValueError("Credentials not found in .env file")
        else:
            print("Credentials loaded successfully.")
        
        try:
            print("Locating email field...")
            email_field = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "username"))
            )
        except Exception:
            print("Using alternative email selector...")
            email_field = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.NAME, "session_key"))
            )
        
        try:
            print("Locating password field...")
            password_field = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "password"))
            )
        except Exception:
            print("Using alternative password selector...")
            password_field = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.NAME, "session_password"))
            )
        
        print("Entering credentials...")
        email_field.clear()
        password_field.clear()
        email_field.send_keys(email)
        if email_field.get_attribute('value') != email:
            raise ValueError("Email input mismatch")
        password_field.send_keys(password)
        
        print("Submitting login form...")
        submit_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']"))
        )
        submit_button.click()
        
        time.sleep(5)
        if "verification" in driver.page_source.lower() or "enter code" in driver.page_source.lower():
            print("Email verification detected! Please complete verification manually in the browser.")
            input("After completing verification, press Enter to continue...")
        
        print("Verifying login success...")
        WebDriverWait(driver, 30).until(
            lambda d: "feed" in d.current_url.lower() or "jobs" in d.current_url.lower()
        )
        print("Login successful!")
        
    except Exception as e:
        driver.save_screenshot("login_error.png")
        snippet = driver.page_source[:1000]
        logging.error(f"Login failed: {str(e)}. Current URL: {driver.current_url}. Page snippet: {snippet}")
        raise

def search_jobs(driver):
    """
    Navigate to the LinkedIn job search page and return a list of job listing elements.
    Uses updated selectors to match LinkedIn's current DOM structure.
    """
    try:
        keywords = "%20".join(config["job_keywords"])
        location = "%20".join(config["locations"])
        url = f"https://www.linkedin.com/jobs/search/?keywords={keywords}&location={location}&f_AL=true"
        print(f"Navigating to job search page: {url}")
        driver.get(url)
        
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.job-card-container--clickable"))
        )
        
        for _ in range(3):
            driver.execute_script("window.scrollBy(0, window.innerHeight)")
            time.sleep(np.random.uniform(1.5, 2.5))
        
        job_elements = driver.find_elements(By.CSS_SELECTOR, "div.job-card-container--clickable")
        print(f"Found {len(job_elements)} job listings.")
        logging.info(f"Found {len(job_elements)} job listings.")
        return job_elements
        
    except Exception as e:
        logging.error(f"Job search failed: {str(e)}")
        driver.save_screenshot("search_error.png")
        raise

def apply_to_job(driver, job_element):
    """
    Applies to a job posting by checking for the Easy Apply feature,
    clicking the Easy Apply button (using both standard and JS click fallback),
    automatically entering the phone number, clicking the "Next" button after phone entry,
    uploading a resume PDF, and navigating through multi-step application forms.
    """
    try:
        # Pre-check: Ensure job has an Easy Apply button
        if not job_has_easy_apply(job_element):
            logging.info("Job does not have Easy Apply feature; skipping.")
            return

        # Click the job element to open details
        job_element.click()
        time.sleep(np.random.uniform(1, 2))
        
        if driver.find_elements(By.XPATH, '//span[contains(text(), "Applied")]'):
            logging.info("Job already applied; skipping.")
            return
        
        print("Looking for Easy Apply button...")
        apply_button = WebDriverWait(driver, 20).until(
            EC.element_to_be_clickable(
                (By.XPATH, '//button[contains(@class, "jobs-apply-button") and contains(., "Easy Apply")]')
            )
        )
        print("Easy Apply Button HTML:", apply_button.get_attribute('outerHTML'))
        driver.execute_script("arguments[0].scrollIntoView(true);", apply_button)
        time.sleep(1)
        
        try:
            logging.info("Attempting normal click on Easy Apply button...")
            apply_button.click()
        except Exception as normal_click_err:
            logging.warning(f"Normal click failed: {str(normal_click_err)}; trying JavaScript click.")
            driver.execute_script("arguments[0].click();", apply_button)
        time.sleep(np.random.uniform(2, 3))
        
        # --- Additional Form Handling: Phone number entry ---
        try:
            phone_field = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '//input[contains(@placeholder, "Phone")]'))
            )
            if phone_field.get_attribute("value") == "":
                phone_field.clear()
                phone_field.send_keys(config["phone_number"])
                logging.info("Entered phone number.")
                time.sleep(1)
        except Exception as phone_err:
            logging.debug(f"Phone field not found: {str(phone_err)}")
        
        # Click "Next" button after phone number entry
        try:
            print("Looking for 'Next' button after phone number entry...")
            next_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//button[contains(@aria-label, "Continue to next step") and contains(., "Next")]'))
            )
            logging.info("Clicking 'Next' button after phone entry...")
            next_button.click()
            time.sleep(np.random.uniform(2, 3))
        except Exception as next_err:
            logging.debug(f"'Next' button after phone entry not found: {str(next_err)}")
        
        # --- Additional Form Handling: Resume PDF Upload ---
        try:
            file_input = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.XPATH, '//input[@type="file"]'))
            )
            resume_pdf_path = os.path.abspath(config["resume_pdf_path"])
            file_input.send_keys(resume_pdf_path)
            logging.info("Uploaded resume PDF.")
            time.sleep(1)
        except Exception as file_err:
            logging.debug(f"File upload field not found: {str(file_err)}")
        # --- End Additional Form Handling ---
        
        # Handle multi-step application process
        max_pages = config.get("max_application_pages", 3)
        for page in range(max_pages):
            try:
                print(f"Looking for Submit button at step {page+1}...")
                submit_button = WebDriverWait(driver, 15).until(
                    EC.element_to_be_clickable((By.XPATH, '//button[contains(@aria-label, "Submit application")]'))
                )
                if submit_button.is_enabled():
                    job_title = driver.find_element(
                        By.XPATH, '//h2[contains(@class, "jobs-details-top-card__job-title")]'
                    ).text
                    logging.info(f"Submitting application for: {job_title}")
                    submit_button.click()
                    time.sleep(np.random.uniform(3, 5))
                    return
            except Exception as submit_err:
                logging.debug(f"Submit button not available on step {page+1}: {str(submit_err)}")
                try:
                    print(f"Looking for Continue button at step {page+1}...")
                    next_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, '//button[contains(@aria-label, "Continue to next step")]'))
                    )
                    logging.info("Clicking 'Continue to next step' button...")
                    next_button.click()
                    time.sleep(np.random.uniform(1, 2))
                except Exception as next_err:
                    logging.error(f"Next button not found on step {page+1}: {str(next_err)}")
                    break
        
        logging.info("Reached end of application steps without automatic submission. Manual intervention required.")
    except Exception as e:
        logging.error(f"Application process failed: {str(e)}")
        driver.save_screenshot("application_error.png")

def main():
    """
    Main execution flow: initializes the driver, logs into LinkedIn,
    searches for jobs, and applies to those with sufficient similarity.
    """
    driver = None
    try:
        print("Initializing browser driver...")
        driver = initialize_driver()
        linkedin_login(driver)
        
        job_elements = search_jobs(driver)
        job_elements = job_elements[:config["max_applications_per_day"]]
        
        for idx, job in enumerate(job_elements):
            try:
                driver.execute_script("arguments[0].scrollIntoView();", job)
                time.sleep(np.random.uniform(0.5, 1.5))
                
                # Attempt to extract job description using two selectors
                job_desc = ""
                try:
                    job_desc_elem = job.find_element(By.CSS_SELECTOR, "div.job-card-container__metadata")
                    job_desc = job_desc_elem.text.strip()
                except Exception:
                    try:
                        job_desc_elem = job.find_element(By.XPATH, ".//div[contains(@class, 'job-card-list__insight')]")
                        job_desc = job_desc_elem.text.strip()
                    except Exception:
                        logging.error("No job description found; skipping this job.")
                        continue
                
                if not job_desc:
                    logging.error("Empty job description; skipping this job.")
                    continue
                
                similarity = job_similarity(job_desc)
                if similarity >= config["min_similarity_score"]:
                    logging.info(f"Processing job {idx+1}/{len(job_elements)} (Similarity Score: {similarity:.2f})")
                    apply_to_job(driver, job)
                    time.sleep(np.random.uniform(10, 20))
                else:
                    logging.info(f"Skipping job {idx+1} due to low similarity (Score: {similarity:.2f})")
            except Exception as job_error:
                logging.error(f"Error processing job {idx+1}: {str(job_error)}")
                continue
                
    except Exception as e:
        logging.error(f"Critical error encountered: {str(e)}")
        if driver:
            driver.save_screenshot("critical_error.png")
    finally:
        if driver:
            driver.quit()
            logging.info("Browser closed successfully.")
        logging.info("Session completed.")

if __name__ == "__main__":
    main()
