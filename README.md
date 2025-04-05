# JobApplicationBot
This repository will help you in applying to jobs in LinkedIn with help of an AI bot.

# In 'config' folder
1. Create a file ".env" and add the follwoing line of codes :-
   LINKEDIN_EMAIL = youremail                  # no quotes
   LINKEDIN_PASSWORD = yourpassword            # no quotes
   HEADLESS = false
   TF_ENABLE_ONEDNN_OPTS = 0

Replace your email and password used in your LinkedIn 

2. In "config.json" update the phone number field with your phone number, update the job keywords associated with your profile and also the your preferred job locations.
   
   {
   "job_keywords": ["Software Intern", "Web Development Intern", "Full Stack Intern", "Front-end Intern", "Back-end Intern"],
    "locations": ["Remote", "Banglore", "Mumbai", "Pune", "Hyderabad", "Delhi", "Gurgaon"],
   "phone_number": "9999999999",
   }

Replace the data according to your requirements.

# In 'data' folder 
1. Upload your resume first in pdf format and delete the pre-existing resume.pdf file.
2. Replace the content of resume.txt file with your resume content.

# While the bot is running:
 * After succesful login it will be asking for verification through your email, perform that manually.
 * The other manual task that you have to perform is of answering the Additional Question, if present in that job profile.

