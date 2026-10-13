import json
import time
from pathlib import Path
import secrets
import string

from pydantic import ValidationError

from prompts import Prompts
from schemas import ClassifierSchema, ResponderSchema, InputEmailSchema, ClassifierOutputSchema, ResponderOutputSchema
from email_history import EmailHistory
from openai_client import LLM
from semantic_search import SemanticSearch
from utils import get_logger


agent_logger = get_logger(name= "customer_agent", filename= "customer_agent.log", level= "debug")



class CustomerAgent:
    #models
    classifier_model = {"name":"gpt-5-mini-2025-08-07",
                        "reasoning" : {"effort":"minimal"},
                        "input_cost": 0.25/1000000,
                        "cached_cost":0.025/1000000,
                        "output_cost": 2/1000000}
    responder_model = {"name":"gpt-5.4-mini-2026-03-17",
                       "reasoning" : {"effort":"medium"},
                        "input_cost": 0.75/1000000,
                        "cached_cost":0.075/1000000,
                        "output_cost": 4.5/1000000}
    
    embedding_model = {"name":"text-embedding-3-small",
                       "query_cost":0.02/1000000,
                       "batch_cost": 0.01/1000000}
    #LLM parameters
    llm_max_retries = 5
    llm_retry_delay = 2
    
    #email history parameters
    history_max_retries = 5
    history_retry_delay= 0.2
    
    #semantic_search parameter
    ss_k_results = 3

    #metrics json path
    metrics_json_path = "metrics.json"
    
    session_id = 0
    sid_prefix = ''.join(secrets.choice(string.ascii_uppercase) for _ in range(4))

    def __init__(self):
        self.classifier_model_name = self.classifier_model["name"]
        self.responder_model_name = self.responder_model["name"]
        self.embedding_model_name =self.embedding_model["name"]
        self.LLM = LLM( max_retries= self.llm_max_retries, 
                       retry_delay= self.llm_retry_delay, 
                       sid_prefix= "LLMG-"+ self.sid_prefix)
        
        self.history = EmailHistory(max_retries= self.history_max_retries, 
                                    retry_delay= self.history_retry_delay, 
                                    sid_prefix = "HIST-"+self.sid_prefix)
        self.prompts = Prompts()
        self.knowledge_base = SemanticSearch(embeddings_model = self.embedding_model_name,
                                             k_results= self.ss_k_results, 
                                             sid_prefix ="KBSS-"+self.sid_prefix)
        self.file = Path(self.metrics_json_path)
        self.metrics_try = 0
        if not self.file.exists():
            self.metrics_try = 1
            self._setup_metrics_file()
        
    def _setup_metrics_file(self):
        data = {
            "semantic_search": {
                "count": 0,
                "model": self.embedding_model,
                "avg_time": 0.0,
                "avg_relevancy": 0.0,
                "avg_query_tokens": 0,
                "total_kb_tokens": 0,
                "avg_cost": 0.0,
                "kb_cost": 0.0,
                "total_cost": 0.0
            },

            "classifier": {
                "count": 0,
                "model": self.classifier_model,
                "avg_time": 0.0,
                "avg_input_tokens": 0,
                "avg_cached_tokens": 0,
                "avg_output_tokens": 0,
                "avg_cost": 0.0,
                "total_cost": 0.0
            },

            "responder": {
                "count": 0,
                "model": self.responder_model,
                "avg_time": 0.0,
                "avg_input_tokens": 0,
                "avg_cached_tokens": 0,
                "avg_output_tokens": 0,
                "avg_cost": 0.0,
                "total_cost": 0.0
            },

            "total_cost": 0.0,
            "total_avg_cost": 0.0,
            "total_avg_time": 0.0,
            "approx_total_cost_per_thousand_replies": 0.0
        }

        self.file.write_text(json.dumps(data, indent=4))
    
    
    def _session_id(self):
        return "AGNT-"+self.sid_prefix+"-"+ str(self.session_id)
    
    def _metrics_update(self, version:str, size: int, time:float)-> None:
        try:
            with open(self.file, "r") as f:
                data = json.load(f)
                if version == "semantic_search" and self.knowledge_base.query_tokens:
                    count = data[version]["count"]
                    data[version]["avg_relevancy"] = (((data[version]["avg_relevancy"] * (count*self.knowledge_base.k_results)) + (self.knowledge_base.RELEVANCY)) / 
                                                    ((count*self.knowledge_base.k_results) + (self.knowledge_base.k_results*size)))
                    
                    data[version]["avg_query_tokens"] = (((data[version]["avg_query_tokens"] * count) + 
                                                        (self.knowledge_base.query_tokens)) / 
                                                        (count + size))
                    
                    data[version]["avg_time"] = (((data[version]["avg_time"] * count) + time) / 
                                                (count + size))
                    data[version]["count"] += size
                    data[version]["total_kb_tokens"] += self.knowledge_base.kb_tokens
                    self.knowledge_base.kb_tokens = 0
                    data[version]["avg_cost"] = (data[version]["avg_query_tokens"] * data[version]["model"]["query_cost"])
                    data[version]["kb_cost"] = (data[version]["total_kb_tokens"] * data[version]["model"]["batch_cost"])
                    data[version]["total_cost"] = (data[version]["avg_cost"] * data[version]["count"]) + data[version]["kb_cost"]
                    

                elif version in ("classifier", "responder"):
                    count = data[version]["count"]
                    if self.LLM.input_token and self.LLM.output_token:
                        data[version]["avg_input_tokens"] = ((data[version]["avg_input_tokens"] * count) + self.LLM.input_token) / (count + size)
                        data[version]["avg_output_tokens"] = ((data[version]["avg_output_tokens"] * count) + self.LLM.output_token) / (count + size)
                        # 0 cached token in a generation is also averaged - to keep average accurate for cost calculations and other performance analysis
                        data[version]["avg_cached_tokens"] = ((data[version]["avg_cached_tokens"] * count) + self.LLM.cached_token) / (count + size)
                        data[version]["avg_time"] = ((data[version]["avg_time"] * count) + time) / (count + size)
                        data[version]["avg_cost"] = ((data[version]["avg_input_tokens"] *  data[version]["model"]["input_cost"]) +
                                                    (data[version]["avg_cached_tokens"] *  data[version]["model"]["cached_cost"]) +
                                                    (data[version]["avg_output_tokens"] *  data[version]["model"]["output_cost"]))
                        data[version]["count"] += size
                        data[version]["total_cost"] = data[version]["avg_cost"] * data[version]["count"]
                else:
                    return
                #aggregating metrics
                data["total_cost"] = ((data["semantic_search"]["total_cost"] + data["classifier"]["total_cost"] + data["responder"]["total_cost"]))
                data["total_avg_cost"] = (data["semantic_search"]["avg_cost"] + data["classifier"]["avg_cost"] + data["responder"]["avg_cost"])
                data["total_avg_time"] = (data["semantic_search"]["avg_time"] + data["classifier"]["avg_time"] + data["responder"]["avg_time"])
                if data["total_cost"] != 0:
                    average_count = (data["semantic_search"]["count"]+ data["classifier"]["count"] + data["responder"]["count"])/3
                    data["approx_total_cost_per_thousand_replies"] = (data["total_cost"]/average_count) *1000

                with open(self.file, "w") as f:
                    json.dump(data, f, indent=4)
        except Exception as e:
            if self.metrics_try == 0:
                self.metrics_try += 1
                self._setup_metrics_file()
                self._metrics_update(version=version,size= size,time=time)
            else: 
                agent_logger.error(f"Session ID: {self._session_id()}   ERROR WHILE UPDATING METRICS FILE \n\n{str(e)}\n\n")

    def classifier(self, email: dict) -> dict:
        self.session_id += 1
        start = time.time()
        try:
            email_address = email.get("email_address")
            history = self.history.get_history( version="classifier", email_address= email_address, limit=1)

            classifier_instruction = self.prompts.classifier_instruction_prompt(email_history= json.dumps(history, indent=4))
            
            classifier_input  = self.prompts.classifier_input_prompt(email= json.dumps(email, indent=4))

            response = self.LLM.generate_response(instructions= classifier_instruction,
                                                input_text= classifier_input,
                                                model_name= self.classifier_model_name, 
                                                response_format=ClassifierSchema,
                                                version="classifier",
                                                reasoning= self.classifier_model["reasoning"])


            validated_response = ClassifierOutputSchema.model_validate({**email,**response}).model_dump()
            
            agent_logger.info("CustomerAgent.classifier success for Session ID: %s", self._session_id())
            
            output = [{"classifier_success" : True}, validated_response] 
            return output
        
        except Exception as e:
            agent_logger.critical("CustomerAgent.classifier failed for Session ID: %s", self._session_id(), exc_info=True)
            output = [{"classifier_success" : False, "error_name": type(e).__name__, "error_detail": str(e)}] 
            return output
        
        finally:
            elapsed = time.time() - start
            self._metrics_update(version="classifier", time=elapsed, size = 1)
        

    

    def search_kb(self, query_text:str)->dict:
        start = time.time()
        try:
            responder_context = self.knowledge_base.query(text= query_text, filename=None)
            return responder_context
        except Exception:
            agent_logger.critical("CustomerAgent.search_kb failed for Session ID: %s", self._session_id(), exc_info=True)
            return {}
        finally:
            elapsed = time.time()-start
            self._metrics_update(version= "semantic_search", time= elapsed, size= 1)




    def responder(self, classifier_data:dict)-> dict:
        start = time.time()
        try:
            query_text = classifier_data["summary"]
            email_address = classifier_data["email_address"]

            email_input = {k: v for k, v in classifier_data.items() if k != "summary"}
            query_result = self.search_kb(query_text=query_text)


            responder_history = self.history.get_history(version= "responder", email_address= email_address, limit = 2)

            responder_instruction = self.prompts.response_instruction_prompt(context= json.dumps(query_result, indent=4),
                                                                              email_history= json.dumps(responder_history, indent=4))
                                                                              
            
            responder_input = self.prompts.response_input_prompt(email= json.dumps(email_input, indent=4))

            response = self.LLM.generate_response(instructions= responder_instruction,
                                                input_text= responder_input,
                                                model_name= self.responder_model_name, 
                                                response_format=ResponderSchema,
                                                version="responder",
                                                reasoning= self.responder_model["reasoning"])
            
            validated_response = ResponderOutputSchema.model_validate({**classifier_data,**response}).model_dump() 
            self.history.add_history(history=validated_response)


            agent_logger.info("CustomerAgent.responder success for Session ID: %s", self._session_id())

            output = [{"responder_success" : True}, validated_response] 
            return output
        except Exception as e:
            agent_logger.critical("CustomerAgent.responder failed for Session ID: %s", self._session_id(), exc_info=True)
            output = [{"responder_success" : False, "error_name": type(e).__name__, "error_detail": str(e)}]
            return output
        
        finally:
            elapsed = time.time()-start
            self._metrics_update(version= "responder", time= elapsed, size= 1)
    
    def reply(self, email:dict)->dict:

        try:
            email = InputEmailSchema.model_validate(email).model_dump()

            classifier_data = self.classifier(email=email)
            if classifier_data[0]["classifier_success"]:
                responder_data = self.responder(classifier_data= classifier_data[1])
                if responder_data[0]["responder_success"]:
                    output= [{"agent_success": True},responder_data[1]]
                else:
                    output = [{"agent_success": False, 
                               "failure" : "CustomerAgent.responder",
                               "error_name":responder_data[0]["error_name"],
                               "error_detail": responder_data[0]["error_detail"]}]      
            else:
                output = [{"agent_success": False, 
                           "failure" : "CustomerAgent.classifier", 
                           "error_name":classifier_data[0]["error_name"],
                           "error_detail": classifier_data[0]["error_detail"]}]
            
        except Exception as error:
            if isinstance(error, ValidationError):
                example_input = InputEmailSchema(email_address= "example@example.com", subject= "Subject Text", body= "Body Text")
                expect = f"\n\nCustomerAgent.reply expects: {example_input.model_dump_json(indent=4)}\n"
                agent_logger.error("CustomerAgent.reply: email input VALIDATION ERROR Session ID: %s \n%s", self._session_id(),str(error)+expect)
                output = [{"agent_success" : False, 
                        "failure" : "CustomerAgent.reply", 
                        "error_name": "InputValidationError",
                        "error_detail": f"Validation failed: Wrong email input format {expect}"}]
            else:
                agent_logger.error("CustomerAgent.reply Failed for Session ID: %s \n\n%s", self._session_id(),str(error))
                output = [{"agent_success" : False, "failure" : "CustomerAgent.reply", "error_name": type(error).__name__,
                        "error_detail": f"\n\n{str(error)}"}]
        finally:
            return output
    def get_response(self, email: dict, retries=3)->dict:
        errors = []
        response = self.reply(email=email)
        for i in range(retries):
            response = self.reply(email=email)
            if response[0]["agent_success"]:
                break
            else:
                time.sleep(5)
                errors.append(response[0])

        if not response[0]["agent_success"]:
            raise RuntimeError(f"ERROR after {retries} retries: \n{json.dumps(errors, indent= 4, default= str)}")
        else:
            return response[1]