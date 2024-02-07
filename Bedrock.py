import boto3
import json 

class Bedrock:
    def __init__(self, modelId, region = 'us-east-1'):
        self.modelId = modelId
        self.region = region
        self.bedrock_runtime = None # boto3.client(service_name = 'bedrock-runtime', region  = self.region)

    def invoke_model(self, body):
        accept = contentType = 'application/json'

        response = self.bedrock_runtime.invoke_model(body=body.encode('utf-8'),
                                                modelId=self.modelId, 
                                                accept=accept, 
                                                contentType=contentType)    

        return json.loads(response.get("body").read())

    
    def get_model_params(self, **kwargs):
        return {
            **{k: v for k, v in kwargs.items() if v is not None}
        }
        

    def build_body(self, input, temperature=None, top_p=None, top_k=None, stop_sequences=None, max_token_count=None, **kwargs):
        model_params_mapping = {
            'amazon.titan-text': self.get_model_params(inputText = input, textGenerationConfig = {
                'temperature': temperature,
                'topP': top_p,
                'stopSequences': stop_sequences,
                'maxTokenCount': max_token_count
             }),            
            'anthropic.claude': self.get_model_params(prompt = input, temperature=temperature, top_p=top_p, top_k=top_k, stop_sequences=stop_sequences, max_tokens_to_sample=max_token_count),
            'ai21': self.get_model_params(prompt = input, temperature=temperature, topP=top_p, stopSequences=stop_sequences, maxTokens=max_token_count, **kwargs),
            'cohere.command-text': self.get_model_params(prompt = input, temperature=temperature, p=top_p, k=top_k, stop_sequences=stop_sequences, max_tokens=max_token_count, **kwargs),
            'meta.llama': self.get_model_params(prompt = input, temperature=temperature, top_p=top_p, max_gen_len=max_token_count)
        }
        
        for model_prefix, model_params in model_params_mapping.items():
            if self.modelId.startswith(model_prefix):
                return model_params
        
        raise ValueError("Unsupported modelId")
    

    def build_prompt(self, prompt, context = None, agent = None, ia_agent=None):
        if agent and ia_agent:
            prompt =  f"{agent}: {prompt} \\n{ia_agent}: "
        if context:
            prompt = f'{prompt} <context> {context} </context>'
        return prompt

def main():
    modelId = 'amazon.titan-text-express-v1'
    bedrock = Bedrock(modelId)
    prompt_data = bedrock.build_prompt("Explain about black holes to a 8th grader", agent = 'Human', ia_agent = 'Assistant')

    # Set inference parameters
    temperature = 0.1
    top_p = 0.9
    max_token_count = 300

    # Get inference parameters
    body = bedrock.build_body(
        input=prompt_data,
        temperature=temperature,
        top_p=top_p,
        max_token_count=max_token_count
        )

    print(modelId)
    print(json.dumps(body))

    modelId = 'anthropic.claude-v2'
    bedrock = Bedrock(modelId)

    # Get inference parameters
    body = bedrock.build_body(
        input=prompt_data,
        temperature=temperature,
        top_p=top_p,
        max_token_count=max_token_count
        )

    print(modelId)
    print(json.dumps(body))






if __name__ == '__main__':
    main()
