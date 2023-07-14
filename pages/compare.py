import hashlib
import uuid
from itertools import cycle

import diff_viewer
import pandas as pd
import requests
import streamlit as st
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from clarifai.listing.lister import ClarifaiResourceLister
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format

st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)


def local_css(file_name):
  with open(file_name) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


local_css("./style.css")

DEBUG = False
completions = []

COHERE = "cohere: generate-base"
OPENAI = "openai: gpt-3.5-turbo"
OPENAI_4 = "openai: gpt-4"
AI21_A = "ai21: j2-jumbo-instruct"
AI21_B = "ai21: j2-grande-instruct"
ANTHROPIC1 = "anthropic: claude-v1"
ANTHROPIC1INSTANT = "anthropic: claude-instant"
ANTHROPIC2 = "anthropic: claude-v2"
# AI21_C = "ai21: j2-jumbo"
# AI21_D = "ai21: j2-grande"
# AI21_E = "ai21: j2-large"

PROMPT_CONCEPT = resources_pb2.Concept(id="prompt", value=1.0)
INPUT_CONCEPT = resources_pb2.Concept(id="input", value=1.0)
COMPLETION_CONCEPT = resources_pb2.Concept(id="completion", value=1.0)

API_INFO = {
    COHERE: {
        "user_id": "cohere",
        "app_id": "generate",
        "model_id": "generate-base",
        "version_id": "07bf79a08a45492d8be5c49085244f1c",
    },
    OPENAI: {
        "user_id": "openai",
        "app_id": "chat-completion",
        "model_id": "GPT-3_5-turbo",
        "version_id": "8ea3880d08a74dc0b39500b99dfaa376",
    },
    OPENAI_4: {
        "user_id": "openai",
        "app_id": "chat-completion",
        "model_id": "GPT-4",
        "version_id": "ad16eda6ac054796bf9f348ab6733c72",
    },
    AI21_A: {
        "user_id": "ai21",
        "app_id": "complete",
        "model_id": "j2-jumbo-instruct",
        "version_id": "d0b0d58b09c947d38bffc0e65b3b1a1b",
    },
    AI21_B: {
        "user_id": "ai21",
        "app_id": "complete",
        "model_id": "j2-grande-instruct",
        "version_id": "620672b5d57043dba8f74d5514cb18ed",
    },
    ANTHROPIC1: {
        "user_id": "anthropic",
        "app_id": "completion",
        "model_id": "claude-v1",
        "version_id": "3a571e774fac465f84d9efcadf0559df",
    },
    ANTHROPIC1INSTANT: {
        "user_id": "anthropic",
        "app_id": "completion",
        "model_id": "claude-instant",
        "version_id": "0363c83d073947d4ba2f76df394dd28d",
    },
    ANTHROPIC2: {
        "user_id": "anthropic",
        "app_id": "completion",
        "model_id": "claude-v2",
        "version_id": "cd8f314bf81f4c24b006af002e827122",
    },
    # AI21_C: {
    #     "user_id": "ai21",
    #     "app_id": "complete",
    #     "model_id": "j2-jumbo",
    #     "version_id": "9bb740d588d743228368a53ac61a3768",
    # },
    # AI21_D: {
    #     "user_id": "ai21",
    #     "app_id": "complete",
    #     "model_id": "j2-grande",
    #     "version_id": "60c292033a4643609b9c553a45f34f24",
    # },
    # AI21_E: {
    #     "user_id": "ai21",
    #     "app_id": "complete",
    #     "model_id": "j2-large",
    #     "version_id": "27122459e3eb44eb9f872afee94d71ae",
    # },
}

# This must be within the display() function.
auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()
lister = ClarifaiResourceLister(stub, auth.user_id, auth.app_id, page_size=16)

st.markdown(
    "<h1 style='text-align: center; color: black;'>LLM Battleground</h1>",
    unsafe_allow_html=True,
)

# st.markdown("Test out LLMs on a variety of tasks. See how they perform!")


def get_user():
  req = service_pb2.GetUserRequest(user_app_id=resources_pb2.UserAppIDSet(user_id="me"))
  response = stub.GetUser(req)
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("GetUser request failed: %r" % response)
  return response.user


user = get_user()
caller_id = user.id


def create_prompt_model(model_id, prompt, position):
  if position not in ["PREFIX", "SUFFIX", "TEMPLATE"]:
    raise Exception("Position must be PREFIX or SUFFIX")

  response = stub.PostModels(
      service_pb2.PostModelsRequest(
          user_app_id=userDataObject,
          models=[
              resources_pb2.Model(
                  id=model_id,
                  model_type_id="prompter",
              ),
          ],
      ))

  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostModels request failed: %r" % response)

  req = service_pb2.PostModelVersionsRequest(
      user_app_id=userDataObject,
      model_id=model_id,
      model_versions=[resources_pb2.ModelVersion(output_info=resources_pb2.OutputInfo())],
  )
  params = json_format.ParseDict(
      {
          "prompt_template": prompt,
          "position": position,
      },
      req.model_versions[0].output_info.params,
  )
  post_model_versions_response = stub.PostModelVersions(req)
  if post_model_versions_response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostModelVersions request failed: %r" % post_model_versions_response)

  return post_model_versions_response.model


def delete_model(model):
  response = stub.DeleteModels(
      service_pb2.DeleteModelsRequest(
          user_app_id=userDataObject,
          ids=[model.id],
      ))
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("DeleteModels request failed: %r" % response)


@st.cache_resource
def create_workflows(prompt, models):
  workflows = []
  prompt_model = create_prompt_model("test-prompt-model-" + uuid.uuid4().hex[:3], prompt,
                                     "TEMPLATE")
  for model in models:
    workflows.append(create_workflow(prompt_model, model))

  st.success(
      f"Created {len(workflows)} workflows! Now ready to test it out by inputting some text below")
  return prompt_model, workflows


def create_workflow(prompt_model, selected_llm):
  req = service_pb2.PostWorkflowsRequest(
      user_app_id=userDataObject,
      workflows=[
          resources_pb2.Workflow(
              id=
              f"test-workflow-{API_INFO[selected_llm]['user_id']}-{API_INFO[selected_llm]['model_id']}-"
              + uuid.uuid4().hex[:3],
              nodes=[
                  resources_pb2.WorkflowNode(
                      id="prompt",
                      model=resources_pb2.Model(
                          id=prompt_model.id,
                          user_id=prompt_model.user_id,
                          app_id=prompt_model.app_id,
                          model_version=resources_pb2.ModelVersion(
                              id=prompt_model.model_version.id,
                              user_id=prompt_model.user_id,
                              app_id=prompt_model.app_id,
                          ),
                      ),
                  ),
                  resources_pb2.WorkflowNode(
                      id="llm",
                      model=resources_pb2.Model(
                          id=API_INFO[selected_llm]["model_id"],
                          user_id=API_INFO[selected_llm]["user_id"],
                          app_id=API_INFO[selected_llm]["app_id"],
                          model_version=resources_pb2.ModelVersion(
                              id=API_INFO[selected_llm]["version_id"],
                              user_id=API_INFO[selected_llm]["user_id"],
                              app_id=API_INFO[selected_llm]["app_id"],
                          ),
                      ),
                      node_inputs=[resources_pb2.NodeInput(node_id="prompt",)],
                  ),
              ],
          ),
      ],
  )

  response = stub.PostWorkflows(req)
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostWorkflows request failed: %r" % response)
  if DEBUG:
    st.json(json_format.MessageToDict(response, preserving_proto_field_name=True))

  return response.workflows[0]


def delete_workflow(workflow):
  response = stub.DeleteWorkflows(
      service_pb2.DeleteWorkflowsRequest(
          user_app_id=userDataObject,
          ids=[workflow.id],
      ))
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("DeleteWorkflows request failed: %r" % response)
  else:
    print(f"Workflow {workflow.id} deleted")


@st.cache_resource
def run_workflow(input_text, workflow):
  response = stub.PostWorkflowResults(
      service_pb2.PostWorkflowResultsRequest(
          user_app_id=userDataObject,
          workflow_id=workflow.id,
          inputs=[
              resources_pb2.Input(
                  data=resources_pb2.Data(text=resources_pb2.Text(raw=input_text,),),),
          ],
      ))
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostWorkflowResults request failed: %r" % response)

  if DEBUG:
    st.json(json_format.MessageToDict(response, preserving_proto_field_name=True))

  return response


@st.cache_resource
def run_model(input_text, model):

  m = API_INFO[model]

  response = stub.PostModelOutputs(
      service_pb2.PostModelOutputsRequest(
          user_app_id=resources_pb2.UserAppIDSet(user_id=m['user_id'], app_id=m['app_id']),
          model_id=m['model_id'],
          inputs=[
              resources_pb2.Input(
                  data=resources_pb2.Data(text=resources_pb2.Text(raw=input_text,),),),
          ],
      ))
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostModelOutputs request failed: %r" % response)

  if DEBUG:
    st.json(json_format.MessageToDict(response, preserving_proto_field_name=True))

  return response


@st.cache_resource
def post_input(txt, concepts=[], metadata=None):
  """Posts input to the API and returns the response."""
  id = hashlib.md5(txt.encode("utf-8")).hexdigest()
  req = service_pb2.PostInputsRequest(
      user_app_id=userDataObject,
      inputs=[
          resources_pb2.Input(
              id=id,
              data=resources_pb2.Data(text=resources_pb2.Text(raw=txt,),),
          ),
      ],
  )
  if len(concepts) > 0:
    req.inputs[0].data.concepts.extend(concepts)
  if metadata is not None:
    req.inputs[0].data.metadata.update(metadata)
  response = stub.PostInputs(req)
  if response.status.code != status_code_pb2.SUCCESS:
    if response.inputs[0].status.details.find("duplicate ID") != -1:
      # If the input already exists, just return the input
      return req.inputs[0]
    raise Exception("PostInputs request failed: %r" % response)
  return response.inputs[0]


def list_concepts():
  """Lists all concepts in the user's app."""
  response = stub.ListConcepts(service_pb2.ListConceptsRequest(user_app_id=userDataObject,))
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("ListConcepts request failed: %r" % response)
  return response.concepts


def post_concept(concept):
  """Posts a concept to the user's app."""
  response = stub.PostConcepts(
      service_pb2.PostConceptsRequest(
          user_app_id=userDataObject,
          concepts=[concept],
      ))
  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("PostConcepts request failed: %r" % response)
  return response.concepts[0]


def search_inputs(concepts=[], metadata=None, page=1, per_page=20):
  """Searches for inputs in the user's app."""
  req = service_pb2.PostAnnotationsSearchesRequest(
      user_app_id=userDataObject,
      searches=[resources_pb2.Search(query=resources_pb2.Query(filters=[]))],
      pagination=service_pb2.Pagination(
          page=page,
          per_page=per_page,
      ),
  )
  if len(concepts) > 0:
    req.searches[0].query.filters.append(
        resources_pb2.Filter(
            annotation=resources_pb2.Annotation(data=resources_pb2.Data(concepts=concepts,))))
  if metadata is not None:
    req.searches[0].query.filters.append(
        resources_pb2.Filter(
            annotation=resources_pb2.Annotation(data=resources_pb2.Data(metadata=metadata,))))
  response = stub.PostAnnotationsSearches(req)
  # st.write(response)

  if response.status.code != status_code_pb2.SUCCESS:
    raise Exception("SearchInputs request failed: %r" % response)
  return response


def get_text(url):
  """Download the raw text from the url"""
  response = requests.get(url)
  return response.text


# Check if prompt, completion and input are concepts in the user's app
app_concepts = list_concepts()
for concept in [PROMPT_CONCEPT, INPUT_CONCEPT, COMPLETION_CONCEPT]:
  if concept.id not in [c.id for c in app_concepts]:
    st.warning(
        f"The {concept.id} concept is not in your app. Please add it by clicking the button below."
    )
    if st.button(f"Add {concept.id} concept"):
      post_concept(concept)
      st.experimental_rerun()

app_concepts = list_concepts()
app_concept_ids = [c.id for c in app_concepts]

# Check if all required concepts are in the app
concepts_ready_bool = True
for concept in [PROMPT_CONCEPT, INPUT_CONCEPT, COMPLETION_CONCEPT]:
  if concept.id not in app_concept_ids:
    concepts_ready_bool = False

# Check if all required concepts are in the app
if not concepts_ready_bool:
  st.error("Need to add all the required concepts to the app before continuing.")
  st.stop()

input_search_response = search_inputs(concepts=[INPUT_CONCEPT], per_page=12)
completion_search_response = search_inputs(concepts=[COMPLETION_CONCEPT], per_page=12)
user_input_search_response = search_inputs(concepts=[INPUT_CONCEPT], per_page=12)

query_params = st.experimental_get_query_params()
inp = ""
if "inp" in query_params:
  inp = query_params["inp"][0]

st.markdown(
    "<h2 style='text-align: center; color: black;'>Try many LLMs at once, see what works best for you</h2>",
    unsafe_allow_html=True,
)

model_names = [OPENAI, OPENAI_4, COHERE, AI21_A, AI21_B, ANTHROPIC1, ANTHROPIC1INSTANT, ANTHROPIC2]
models = st.multiselect(
    "Select the LLMs you want to use:", model_names, default=[OPENAI_4, ANTHROPIC2])

inp = st.text_area(
    " ",
    placeholder="Send a message to the LLMs",
    value=inp,
    help="Genenerate outputs from the LLMs using this input.")

if inp and models:
  if len(models) == 0:
    st.error("You need to select at least one model.")
    st.stop()

  concepts = list_concepts()
  concept_ids = [c.id for c in concepts]
  for concept in [INPUT_CONCEPT, COMPLETION_CONCEPT]:
    if concept.id not in concept_ids:
      post_concept(concept)
      st.success(f"Added {concept.id} concept")

  inp_input = post_input(
      inp,
      concepts=[INPUT_CONCEPT],
      metadata={"tags": ["input"],
                "caller": caller_id},
  )

  st.markdown(
      "<h1 style='text-align: center;font-size: 40px;color: #667085;'>Completions</h1>",
      unsafe_allow_html=True,
  )

  cols = st.columns(3)

  for mi, model in enumerate(models):
    col = cols[mi % len(cols)]
    container = col.container()
    prediction = run_model(inp, model)
    m = API_INFO[model]
    h = ClarifaiUrlHelper(auth)
    model_url = h.clarifai_url(m["user_id"], m["app_id"], "models", m["model_id"])
    model_url_with_version = h.clarifai_url(m["user_id"], m["app_id"], "models", m["model_id"],
                                            m["version_id"])
    container.write(f"Completion from {model_url}:")

    completion = prediction.outputs[0].data.text.raw
    container.write(completion)
    complete_input = post_input(
        completion,
        concepts=[COMPLETION_CONCEPT],
        metadata={
            "input_id": inp_input.id,
            "tags": ["completion"],
            "model": model_url_with_version,
            "caller": caller_id,
        },
    )
    completions.append({
        "select": True,
        "model": model_url,
        "completion": completion.strip(),
        # "input_id":
        #     f"https://clarifai.com/{userDataObject.user_id}/{userDataObject.app_id}/inputs/{complete_input.id}",
    })

  c = pd.DataFrame(completions)

  st.subheader("Show differences")
  st.markdown("Select the completions you wish to compare.")
  edited_df = st.data_editor(c, disabled=set(completions[0].keys()) - set(["select"]))
  selected_rows = edited_df.loc[edited_df['select']]
  if len(selected_rows) != 2:
    st.warning("Please select two completions to diff")
  else:
    old_value = selected_rows.iloc[0]["completion"]
    new_value = selected_rows.iloc[1]["completion"]
    cols = st.columns(2)
    cols[0].markdown(f"Completion: {selected_rows.iloc[0]['model']}")
    cols[1].markdown(f"Completion: {selected_rows.iloc[1]['model']}")
    diff_viewer.diff_viewer(old_text=old_value, new_text=new_value, lang='none')

st.markdown(
    "Note: your messages and completions will be stored and shares publicly as recent messages")

st.button("Request a feature in this module")
st.button("Learn how to build your own module")
st.button("See how this module was built")
st.button("Share these generations")

with st.expander("Recent Messages from Others"):

  st.markdown(
      "<div style='text-align: center;'>Hover to copy and try them out yourself!</div>",
      unsafe_allow_html=True,
  )

  @st.cache_resource
  def completions_for_input(input_id):
    completions = []
    for completion_hit in completion_search_response.hits:
      if completion_hit.input.data.metadata.fields["input_id"].string_value == input_id:
        txt = get_text(completion_hit.input.data.text.url)
        model_url = completion_hit.input.data.metadata.fields["model"].string_value
        completions.append((txt, model_url))
    return completions

  previous_inputs = []

  st.json(json_format.MessageToJson(completion_search_response, preserving_proto_field_name=True))

  cols = cycle(st.columns(3))
  for idx, input_hit in enumerate(input_search_response.hits):
    txt = get_text(input_hit.input.data.text.url)
    previous_inputs.append({
        "input": txt,
    })
    container = next(cols).container()
    metadata = json_format.MessageToDict(input_hit.input.data.metadata)
    caller_id = metadata.get("caller", "zeiler")
    if caller_id == "":
      caller_id = "zeiler"

    container.subheader(f"Input ({caller_id})", anchor=False)
    container.code(txt)  # metric(label="Input", value=txt)

    container.subheader("Completions:", anchor=False)

    completions = completions_for_input(input_hit.input.id)

    container.write(completions)

st.markdown(
    "<h3 style='text-align: center; color: black;'>Built on Clarifai with 💙 </h3>",
    unsafe_allow_html=True,
)
