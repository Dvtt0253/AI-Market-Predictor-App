import time 
import threading
from collections import defaultdict
from flask import request
from datetime import datetime, timedelta
import bleach
import secrets 
import logging

# Copyright (c) 2025 Joie Harvey
# All Rights Reserved.
#
# Licensed under the All Rights Reserved. Unauthorized use or redistribution is prohibited.



logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='firewall_log.log',
    filemode= 'w'


)

tmpblacklist_csv = 'tempblacklist.csv'
permblacklist_csv = 'permblacklist.csv'
whitelist_csv = 'whitelist_csv'











class Firewall:
    def __init__(self, max_requests, time_window):
        self.requests = defaultdict(list)
        self.max_requests = max_requests
        self.time_window = time_window
        self.permanent_blacklist = {}
        self.temporary_blacklist = {}
        self.whitelist = {}
        self.login_requests = defaultdict(list)
        self.secret_key = secrets.token_hex(16) 
        self.violations = {}
        self.lock = threading.Lock()


    

        
#  Rate Limiting and Login attempts limiting 
    def rate_limiter(self):
        with self.lock:
            current_time = time.time()
            current_date = datetime.now()
            ip_address = request.remote_addr
            if ip_address in self.whitelist:
                return 200
            self.requests[ip_address].append(current_time)
            recent_requests = [timestamp for timestamp in self.requests[ip_address] if current_time - timestamp <= self.time_window ]

            if len(recent_requests) > self.max_requests:
                if ip_address not in self.temporary_blacklist:
                    self.temporary_blacklist[ip_address] = current_date
                    
                    
                    logging.info(f"{ip_address} has been temporarily blacklisted due to surpassing the rate limit.")
                
                return 429
            else:
                return 200
            
    def login_limiter(self, max_attempts, time_window):
        with self.lock:
            
            current_time = time.time()
            ip_address = request.remote_addr
            if ip_address in self.whitelist:
                return 200
            current_date = datetime.now()
            
            
            self.login_requests[ip_address].append(current_time)
            recent_attempts = [timestamp for timestamp in self.login_requests[ip_address] if current_time - timestamp <= time_window]

            if len(recent_attempts) > max_attempts:
                if ip_address not in self.temporary_blacklist:
                    self.temporary_blacklist[ip_address] = current_date
                   
                    
                    logging.info(f"{ip_address} has been temporarily blacklisted due to too many failed login attempts.")

            
                return 403
            else:
                return 200
        

        




# Input handling and sanitization 
    def santitize_input(self, user_input):
        sanitized_input = bleach.clean(user_input, tags=[], attributes=[], strip=True)
        return sanitized_input

        
   
        
    def identify_payloads(self, request_data):
        with self.lock:
            ip_address = request.remote_addr

            current_date = datetime.now()


            
            with open('payloads.txt', "r") as data:
                for line in data:
                    if request_data.strip() == line.strip():
                        self.permanent_blacklist[ip_address] = (current_date, "Malicious use of payloads.")
                        
                        logging.info(f"{ip_address} has been permanently blacklisted due to attempting to inject malicious payloads.")

                        return 403
                    
                return 200
                    

                
    


                
#  Access control and Blacklist handling 
    def block_access(self):
        with self.lock:
            ip_address = request.remote_addr
            if ip_address in self.permanent_blacklist or ip_address in self.temporary_blacklist:
                return  403
            else:
                return  200


    def removeFrom_tempBlacklist(self):
      
            while True:
                removed_ips = []
                with self.lock:

                    for key, date in list(self.temporary_blacklist.items()):
                        removal_date = date + timedelta(minutes = 30)
                        if datetime.now() >= removal_date:
                            removed_ips.append(key)



                for ip in removed_ips:
                    del(self.temporary_blacklist[ip])
                   
                   
                    logging.info(f"{ip} has been removed from the temporary blacklist.")

                time.sleep(60)

    def startTempBlacklist_removal(self):
        thread = threading. Thread(target=self.removeFrom_tempBlacklist)
        thread.start()
    
    def is_blacklisted(self, ip_address):
        with self.lock:
            if ip_address in self.temporary_blacklist or ip_address in self.permanent_blacklist:
                return True
            else:
                return False 
            

    def IP_violation_count(self):
        
            while True:
                time.sleep(1798)
                with self.lock:
                    for ip_address in self.temporary_blacklist:
                        if ip_address in self.violations:

                            self.violations[ip_address] += 1
                            logging.info(f"{ip_address} has a violations count of {self.violations[ip_address]}")
                        else:
                            self.violations[ip_address] = 1
                            logging.info(f"{ip_address} has a violations count of {self.violations[ip_address]}")
                   
                    
                        
                        

            

# checks if IP address is in temporary blacklist every 28 mins and updates the IP violation count

    def startperiodic_check(self):
         logging.info("Violations tracker has begun...")
         thread = threading.Thread(target=self.IP_violation_count)
         thread.start()

   

        
        
# Permanently blacklists Ips that have been temporarily blacklisted a certain amount of times
    def temp_blacklist_threshold(self):
        with self.lock:
            current_date = datetime.now()
            threshold = 3
            temp_to_perm = []

            
            for ip_address, violation_count in list(self.violations.items()):
                if violation_count >= threshold:

                    self.permanent_blacklist[ip_address] = (current_date, "Violations threshold has been met.")
                   
                    logging.info(f"{ip_address} has been permanently blacklisted due to repeated violations.")
                    temp_to_perm.append(ip_address)
                
            for ip in temp_to_perm:
                del(self.violations[ip])

            

    
    
    def manual_temp_removal(self, ip_address):
        with self.lock:
            if ip_address in self.temporary_blacklist:
                del(self.temporary_blacklist[ip_address])
               
                
                logging.info(f"{ip_address} has been removed from temporary blacklist.")


   
        






    
#  Whitelist Handling 
    def add_to_whitelist(self, ip_address, actions):
        with self.lock:

            if ip_address not in self.whitelist:

                self.whitelist[ip_address] = actions
                
                
                logging.info(f"{ip_address} has been whitelisted.")

                
            else:
                return  400


    def remove_from_whitelist(self, ip_address):
        with self.lock:
            if ip_address in self.whitelist:

                del(self.whitelist[ip_address])
               
                
                logging.info(f"{ip_address} has been removed from whitelist.")
               
            else:
                return 400
            
    def is_whitelisted(self, ip_address):
        with self.lock:
            if ip_address in self.whitelist:
                return True
            else:
                return False
        
        

        






#  CSRF Protections 


    def generate_CSRF_Token(self, length):

            return secrets.token_hex(length)


    def validate_CSRF(original_CSRF, entered_CSRF):
        if original_CSRF == entered_CSRF:
            return  200
        else:
            return  400





    
        
           
            

            




           


        
        
