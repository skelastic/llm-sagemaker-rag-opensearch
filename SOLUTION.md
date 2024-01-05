
## Solution Overview

In this post we provide a step-by-step guide with all the building
blocks for creating an enterprise ready RAG application such as a
question answering bot. We use a combination of different AWS services,
open-source foundation models ([FLAN-T5
XXL](https://huggingface.co/google/flan-t5-xxl) for text generation and
[GPT-j-6B](https://huggingface.co/EleutherAI/gpt-j-6b) for embeddings)
and packages such as
[LangChain](https://python.langchain.com/en/latest/index.html) for
interfacing with all the components and
[Streamlit](https://streamlit.io/) for building the bot frontend.

- Interfacing with LLMs hosted on SageMaker.
- Chunking of knowledge base documents.
- Ingesting document embeddings into an OpenSearch cluster.
- Implementing the question answering task.

We can use the same architecture to swap the open-source models with the
[Amazon Titan](https://aws.amazon.com/bedrock/titan/) models. After
[Amazon Bedrock](https://aws.amazon.com/bedrock/) launches, we will
publish a follow-up post showing how to implement similar generative AI
applications using Amazon Bedrock, so stay tuned.

The [SageMaker docs](https://sagemaker.readthedocs.io) are converted 
into smaller overlapping chunks (to retain some context continuity
between chunks) of information and then convert these chunks into
embeddings using the gpt-j-6b model and store the embeddings in
an OpenSearch cluster. We implement the RAG functionality inside an AWS
Lambda function with Amazon API Gateway to handle routing all requests
to the Lambda. We implement A chatbot application using Streamlit framework invokes a Lambda function via an API Gateway and the function performs a similarity search in the OpenSearch embeddings index. The matching documents (chunks) are added to the prompt
as context by the Lambda function and then the function uses the
flan-t5-xxl model deployed as a SageMaker endpoint to generate an answer to the input/query.

Step-by-step explanation:

1.  The User provides a question via the Streamlit web application.
2.  The Streamlit application invokes the API Gateway endpoint REST API.
3.  The API Gateway invokes the Lambda function.
4.  The function invokes the SageMaker endpoint to convert user question
    into embeddings.
5.  The function invokes invokes an OpenSearch Service API to find
    similar documents to the user question.
6.  The function creates a “prompt” with the user query and the “similar
    documents” as context and asks the SageMaker endpoint to generate a
    response.
7.  The response is provided from the function to the API Gateway.
8.  The API Gateway provides the response to the Streamlit application.
9.  The User is able to view the response on the Streamlit application,

As illustrated in the architecture diagram, we use the following AWS
services:

- [SageMaker](https://aws.amazon.com/pm/sagemaker) and [Amazon SageMaker
  JumpStart](https://aws.amazon.com/sagemaker/jumpstart/) for hosting
  the two LLMs.
- [OpenSearch Service](https://aws.amazon.com/opensearch-service/) for
  storing the embeddings of the enterprise knowledge corpus and doing
  similarity search with user questions.
- [Lambda](https://aws.amazon.com/lambda/) for implementing the RAG
  functionality and exposing it as a REST endpoint via the [API
  Gateway](https://aws.amazon.com/api-gateway/).
- [Amazon SageMaker Processing
  jobs](https://docs.aws.amazon.com/sagemaker/latest/dg/processing-job.html)
  for large scale data ingestion into OpenSearch.
- [Amazon SageMaker Studio](https://aws.amazon.com/sagemaker/studio/)
  for hosting the Streamlit application.
- [AWS Identity and Access Management](https://aws.amazon.com/iam/)
  roles and policies for access management.
- [AWS CloudFormation](https://aws.amazon.com/cloudformation/) for
  creating the entire solution stack through infrastructure as code.

In terms of open-source packages used in this solution, we use
[LangChain](https://python.langchain.com/en/latest/index.html) for
interfacing with OpenSearch Service and SageMaker, and
[FastAPI](https://github.com/tiangolo/fastapi) for implementing the REST
API interface in the Lambda.

The workflow for instantiating the solution presented in this post in
your own AWS account is as follows:

1.  Run the CloudFormation template provided with this post in your
    account. This will create all the necessary infrastructure resources
    needed for this solution:

    1.  SageMaker endpoints for the LLMs
    2.  OpenSearch Service cluster
    3.  API Gateway
    4.  Lambda function
    5.  SageMaker Notebook
    6.  IAM roles

2.  Run the
    [`data_ingestion_to_vectordb.ipynb`](./data_ingestion_to_vectordb.ipynb)
    notebook in the SageMaker notebook to ingest data from [SageMaker
    docs](https://sagemaker.readthedocs.io) into an OpenSearch Service
    index.

3.  Run the Streamlit application on a terminal in Studio and open the
    URL for the application in a new browser tab.

4.  Ask questions about SageMaker via the chat interface provided
    by the Streamlit app and view the responses generated by the LLM.

These steps are discussed in detail in the following sections.

### Prerequisites

To implement the solution provided in this post, you should have an [AWS
account](https://signin.aws.amazon.com/signin?redirect_uri=https%3A%2F%2Fportal.aws.amazon.com%2Fbilling%2Fsignup%2Fresume&client_id=signup)
and familiarity with LLMs, OpenSearch Service and SageMaker.

We need access to accelerated instances (GPUs) for hosting the LLMs.
This solution uses one instance each of `ml.g5.12xlarge` and
`ml.g5.24xlarge`; you can check the availability of these instances in
your AWS account and request these instances as needed via a
`Sevice Quota` increase request as shown in the following screenshot.


#### Use AWS Cloud Formation to create the solution stack

After the stack is created successfully, navigate to the stack’s
`Outputs` tab on the AWS CloudFormation console and note the values for
`OpenSearchDomainEndpoint` and `LLMAppAPIEndpoint`. We use those in the
subsequent steps.


#### Ingest the data into OpenSearch Service

To ingest the data, complete the following steps:

1.  On the SageMaker console, choose **Notebooks** in the navigation
    pane.

2.  Select the notebook aws-llm-apps-blog and choose **Open
    JupyterLab**.

3.  Choose
    [`data_ingestion_to_vectordb.ipynb`](./data_ingestion_to_vectordb.ipynb)
    to open it in JupyterLab. This notebook will ingest the [SageMaker
    docs](https://sagemaker.readthedocs.io) to an OpenSearch Service
    index called `llm_apps_workshop_embeddings`.

4.  When the notebook is open, on the Run menu, choose **Run All Cells**
    to run the code in this notebook. This will download the dataset
    locally into the notebook and then ingest it into the OpenSearch
    Service index. This notebook takes about 20 minutes to run. The
    notebook also ingests the data into another vector database called
    [`FAISS`](https://github.com/facebookresearch/faiss). The FAISS
    index files are saved locally and the uploaded to Amazon Simple
    Storage Service (S3) so that they can optionally be used by the
    Lambda function as an illustration of using an alternate vector
    database.

    Now we’re ready to split the documents into chunks, which can then
    be converted into embeddings to be ingested into OpenSearch. We use
    the LangChain `RecursiveCharacterTextSplitter` class to chunk the
    documents and then use the LangChain
    `SagemakerEndpointEmbeddingsJumpStart` class to convert these chunks
    into embeddings using the gpt-j-6b LLM. We store the embeddings in
    OpenSearch Service via the LangChain `OpenSearchVectorSearch` class.
    We package this code into Python scripts that are provided to the
    SageMaker Processing Job via a custom container. See the
    [`data_ingestion_to_vectordb.ipynb`](https://github.com/aws-samples/llm-apps-workshop/blob/main/blogs/rag/data_ingestion_to_vectordb.ipynb)
    notebook for the full code.

    1.  Create a custom container, then install in it the `LangChain`
        and `opensearch-py` Python packages.
    2.  Upload this container image to Amazon Elastic Container Registry
        (ECR).
    3.  We use the SageMaker `ScriptProcessor` class to create a
        SageMaker Processing job that will run on multiple nodes.
        - The data files available in Amazon S3 are automatically
          distributed across in the SageMaker Processing job instances
          by setting `s3_data_distribution_type='ShardedByS3Key'` as
          part of the `ProcessingInput` provided to the processing job.
        - Each node processes a subset of the files and this brings down
          the overall time required to ingest the data into OpenSearch
          Service.
        - Each node also uses Python `multiprocessing` to internally
          also parallelize the file processing. Therefore, **there are
          two levels of parallelization happening, one at the cluster
          level where individual nodes are distributing the work (files)
          amongst themselves and another at the node level where the
          files in a node are also split between multiple processes
          running on the node**.

    ``` python
    # setup the ScriptProcessor with the above parameters
    processor = ScriptProcessor(base_job_name=base_job_name,
                                image_uri=image_uri,
                                role=aws_role,
                                instance_type=instance_type,
                                instance_count=instance_count,
                                command=["python3"],
                                tags=tags)

    # setup input from S3, note the ShardedByS3Key, this ensures that 
    # each instance gets a random and equal subset of the files in S3.
    inputs = [ProcessingInput(source=f"s3://{bucket}/{app_name}/{DOMAIN}",
                              destination='/opt/ml/processing/input_data',
                              s3_data_distribution_type='ShardedByS3Key',
                              s3_data_type='S3Prefix')]


    logger.info(f"creating an opensearch index with name={opensearch_index}")
    # ready to run the processing job
    st = time.time()
    processor.run(code="container/load_data_into_opensearch.py",
                  inputs=inputs,
                  outputs=[],
                  arguments=["--opensearch-cluster-domain", opensearch_domain_endpoint,
                            "--opensearch-secretid", os_creds_secretid_in_secrets_manager,
                            "--opensearch-index-name", opensearch_index,
                            "--aws-region", aws_region,
                            "--embeddings-model-endpoint-name", embeddings_model_endpoint_name,
                            "--chunk-size-for-doc-split", str(CHUNK_SIZE_FOR_DOC_SPLIT),
                            "--chunk-overlap-for-doc-split", str(CHUNK_OVERLAP_FOR_DOC_SPLIT),
                            "--input-data-dir", "/opt/ml/processing/input_data",
                            "--create-index-hint-file", CREATE_OS_INDEX_HINT_FILE,
                            "--process-count", "2"])
    ```

5.  Close the notebook after all cells run without any error. Your data
    is now available in OpenSearch Service. Enter the following URL in
    your browser’s address bar to get a count of documents in the
    `llm_apps_workshop_embeddings` index. Use the OpenSearch Service
    domain endpoint from the CloudFormation stack outputs in the URL
    below. You’d be prompted for the OpenSearch Service username and
    password, these are available from the CloudFormations stack.

        https://<your-opensearch-domain-endpoint>/llm_apps_workshop_embeddings/_count

    The browser window should show an output similar to the following.
    This output shows that 5,667 documents were ingested into the
    `llm_apps_workshop_embeddings` index.
    `{"count":5667,"_shards":{"total":5,"successful":5,"skipped":0,"failed":0}}`

### Run the Streamlit application in Studio

Now we’re ready to run the Streamlit web application for our question
answering bot. This application allows the user to ask a question and
then fetches the answer via the `/llm/rag` REST API endpoint provided by
the Lambda function.

Studio provides a convenient platform to host the Streamlit web
application. The following steps describes how to run the Streamlit app
on Studio. Alternatively, you could also follow the same procedure to
run the app on your laptop.

1.  Open Studio and then open a new terminal.

2.  Run the following commands on the terminal to clone the code
    repository for this post and install the Python packages needed by
    the application:

    ``` bash
    git clone https://github.com/aws-samples/llm-apps-workshop
    cd llm-apps-workshop/blogs/rag/app
    pip install -r requirements.txt
    ```

3.  The API Gateway endpoint URL that is available from the
    CloudFormation stack output needs to be set in the webapp.py file.
    This is done by running the following `sed` command. Replace the
    `<replace-with-LLMAppAPIEndpoint-value-from-cloudformation-stack-outputs>`
    in the shell commands with the value of the `LLMAppAPIEndpoint`
    field from the CloudFormation stack output and then run the
    following commands to start a Streamlit app on Studio.

    ``` bash
    EP=<replace-with-LLMAppAPIEndpoint-value-from-cloudformation-stack-outputs>
    # replace __API_GW_ENDPOINT__ with  output from the cloud formation stack
    sed -i "s|__API_GW_ENDPOINT__|$EP|g" webapp.py
    streamlit run webapp.py    
    ```

4.  When the application runs successfully, you’ll see an output similar
    to the following (the IP addresses you will see will be different
    from the ones shown in this example). **Note the port number
    (typically 8501) from the output** to use as part of the URL for app
    in the next step.

    ``` bash
    sagemaker-user@studio$ streamlit run webapp.py 

    Collecting usage statistics. To deactivate, set browser.gatherUsageStats to False.


      You can now view your Streamlit app in your browser.

      Network URL: http://169.255.255.2:8501
      External URL: http://52.4.240.77:8501
    ```

5.  You can access the app in a new browser tab using a URL that is
    similar to your Studio domain URL. For example, if your Studio URL
    is
    `https://d-randomidentifier.studio.us-east-1.sagemaker.aws/jupyter/default/lab?`
    then the URL for your Streamlit app will be
    `https://d-randomidentifier.studio.us-east-1.sagemaker.aws/jupyter/default/proxy/8501/webapp`


### A closer look at the RAG implementation in the Lambda function

Now that we have the application working end to end, lets take a closer
look at the Lambda function. The Lambda function uses
[`FastAPI`](https://fastapi.tiangolo.com/lo/) to implement the REST API
for RAG and the [`Mangum`](https://pypi.org/project/mangum/) package to
wrap the API with a handler that we package and deploy in the function.
We use the API Gateway to route all incoming requests to invoke the
function and handle the routing internally within our application.

The following code snippet shows how we find documents in the OpenSearch
index that are similar to the user question and then create a prompt by
combining the question and the similar documents. This prompt is then
provided to the LLM for generating an answer to the user question.

``` python

@router.post("/rag")
async def rag_handler(req: Request) -> Dict[str, Any]:
    # dump the received request for debugging purposes
    logger.info(f"req={req}")

    # initialize vector db and SageMaker Endpoint
    _init(req)

    # Use the vector db to find similar documents to the query
    # the vector db call would automatically convert the query text
    # into embeddings
    docs = _vector_db.similarity_search(req.q, k=req.max_matching_docs)
    logger.info(f"here are the {req.max_matching_docs} closest matching docs to the query=\"{req.q}\"")
    for d in docs:
        logger.info(f"---------")
        logger.info(d)
        logger.info(f"---------")

    # now that we have the matching docs, lets pack them as a context
    # into the prompt and ask the LLM to generate a response
    prompt_template = """Answer based on context:\n\n{context}\n\n{question}"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    logger.info(f"prompt sent to llm = \"{prompt}\"")
    chain = load_qa_chain(llm=_sm_llm, prompt=prompt)
    answer = chain({"input_documents": docs, "question": req.q}, return_only_outputs=True)['output_text']
    logger.info(f"answer received from llm,\nquestion: \"{req.q}\"\nanswer: \"{answer}\"")
    resp = {'question': req.q, 'answer': answer}
    if req.verbose is True:
        resp['docs'] = docs

    return resp
```
