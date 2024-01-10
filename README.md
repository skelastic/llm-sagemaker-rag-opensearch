# QA with LLM and RAG

CloudFormation Deployment:

   |AWS Region                |     Link        |
   |:------------------------:|:-----------:|
   |us-east-1   | [CloudFormation Template](https://console.aws.amazon.com/cloudformation/home?region=us-east-1#/stacks/new?stackName=llm-sagemaker-rag-opensearch&templateURL=https://raw.githubusercontent.com/skelastic/llm-sagemaker-rag-opensearch/main/rag-opensearch-sagemaker.yaml) |
   |us-west-2       | [CloudFormation Template](https://console.aws.amazon.com/cloudformation/home?region=us-west-2#/stacks/new?stackName=llm-sagemaker-rag-opensearch&templateURL=https://raw.githubusercontent.com/skelastic/llm-sagemaker-rag-opensearch/main/rag-opensearch-sagemaker.yaml) |
   
1. Once the cloud formation stack has been created successfully, open the Outputs tab of the stack and note the URL for the API Gateway we will be needing it to the run a RAG query later on.

1. Open the `rag-opensearch-sagemaker` SageMaker Notebook created by the deployment and then open the [`data_ingestion_to_vectordb.ipynb`](data_ingestion_to_vectordb.ipynb) file.

1. It will ingest the data (embeddings) into the OpenSearch cluster and once that is done, we are now ready to ask some questions via the `/rag` endpoint of the Lambda function.

1. Query the API Gateway `/rag` endpoint using the following command. The endpoint can be seen on the Outputs tab of the cloud formation stack, it is value of the `LLMAppAPIEndpoint` key.

    ```{{bash}}
    curl -X POST "https://API-ENDPOINT-URL/prod/api/v1/llm/rag" -H  "accept: application/json" -H  "Content-Type: application/json" -d "{\"q\":\"Which versions of XGBoost does SageMaker support?\"}"
    ```
1. Run the [`streamlit`](https://streamlit.io/) app for the chatbot on `SageMaker Studio`. On `Sagemaker Studio` create a new `Terminal` and run the following commands:

    ```{{bash}}
    git clone https://github.com/skelastic/llm-sagemaker-rag-opensearch
    cd app
    pip install -r requirements.txt
    streamlit run webapp.py    
    ```
    This will start a streamlit app on SageMaker Studio, you can access the app by opening the following URL in a new browser tab `https://SAGEMAKER-STUDIO-URL/jupyter/default/proxy/8501/webapp`
