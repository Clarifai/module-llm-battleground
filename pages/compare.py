import hashlib
import time
import uuid
from itertools import cycle
from typing import Dict, Iterator

import diff_viewer
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.auth import V2Stub, create_stub
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai.client.model import Model #New import to support list_model function
from clarifai.client.app import App     #New import to support list_model function
from clarifai.client.input import Inputs
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Struct
from google.protobuf.json_format import MessageToDict

st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)


if 'generated_completions' not in st.session_state:
  st.session_state['generated_completions'] = False


def local_css(file_name):
  with open(file_name) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


def load_pat():
  if 'CLARIFAI_PAT' not in st.secrets:
    st.error("You need to set the CLARIFAI_PAT in the secrets.")
    st.stop()
  return st.secrets.CLARIFAI_PAT


def get_default_models():
  if 'DEFAULT_MODELS' not in st.secrets:
    st.error("You need to set the default models in the secrets.")
    st.stop()
  models = st.secrets.DEFAULT_MODELS.split(",")
  return models


def get_userapp_scopes(stub: V2Stub, userDataObject):
  userDataObj = resources_pb2.UserAppIDSet(
      user_id=userDataObject.user_id, app_id=userDataObject.app_id)
  response = stub.MyScopes(service_pb2.MyScopesRequest(user_app_id=userDataObj))
  return response


def validate_scopes(required_scopes, userapp_scopes):
  if "All" in userapp_scopes or all(scp in userapp_scopes for scp in required_scopes):
    return True
  return False


def show_error(request_name, response):
  st.error(f"There was an error with your request to {request_name}")
  st.json(json_format.MessageToJson(response, preserving_proto_field_name=True))
  raise Exception(
      f"There was an error with your request to {request_name} {response.status.description}")
  st.stop()


def list_all_models(
    filter_by: dict = {},
) -> Dict[str, Dict[str, str]]:
  """
    Iterator for all the LLM community models.

    Args:
      filter_by: a dictionary of filters to apply to the list of models.

    Returns:
      API_INFO: dictionary of models information.
    """  
  llm_community_models = App().list_models(filter_by=filter_by,
                                                         only_in_app=False)
  API_INFO={}
  for model_name in llm_community_models:
    model_dict=MessageToDict(model_name.model_info)
    try:
      API_INFO[f"{model_dict['id']}: {model_dict['userId']}"] = dict(
          user_id=model_dict["userId"],
          app_id=model_dict["appId"],
          model_id=model_dict["id"],
          version_id=model_dict["modelVersion"]["id"])
    except IndexError:
      pass
  return API_INFO 


local_css("./style.css")

DEBUG = False
completions = []

PROMPT_CONCEPT = resources_pb2.Concept(id="prompt", value=1.0)
INPUT_CONCEPT = resources_pb2.Concept(id="input", value=1.0)
COMPLETION_CONCEPT = resources_pb2.Concept(id="completion", value=1.0)

# This must be within the display() function.

# We need to use this post_input and to create and delete models/workflows.
secrets_auth = ClarifaiAuthHelper.from_streamlit(st)
pat = load_pat()
secrets_auth._pat = pat
secrets_stub = create_stub(secrets_auth)  # installer's stub (PAT)

# TODO(mansi): validate with myscopes that we have all the scopes we need for the API calls to post
# inputs and delete models/workflows.

# Check if the user is logged in. If not, use internal PAT.
module_query_params = st.experimental_get_query_params()
if module_query_params.get("pat", "") == "" and module_query_params.get("token", "") == "":
  unauthorized = True
else:
  unauthorized = False
# Get the auth from secrets first and then override that if a pat is provided as a query param.
# If no PAT is in the query param then the resulting auth/stub will match the secrets_auth/stub.
user_or_secrets_auth = ClarifaiAuthHelper.from_streamlit(st)
# This user_or_secrets_stub wil be used for all the predict calls so we bill the user for those.
user_or_secrets_stub = create_stub(user_or_secrets_auth)  # user's (viewer's) stub
userDataObject = user_or_secrets_auth.get_user_app_id_proto()

# We are using user's (viewer's) PAT for ListModelsRequest, PostWorkflowResults & PostModelOutputs
# For other API calls we are using installer's PAT from secrets.toml file
# So I am checking scopes on user's key  - only those scopes which are required for ListModelsRequest, PostWorkflowResults & PostModelOutputs

needed_read_scopes = ['Inputs:Get', 'Models:Get', 'Concepts:Get', 'Predict', 'Workflows:Get']
myscopes_response = get_userapp_scopes(user_or_secrets_stub, userDataObject)
if not validate_scopes(needed_read_scopes, myscopes_response.scopes):
  st.error("You do not have correct scopes for this module")
  st.stop()

# If the user has these write scopes, it means that the user has installed this module in his app. 
# Note: this step is necessary as in secrets.toml, we have PAT for from Clarifai org's account which won't have write scopes for user's installed module.
needed_write_scopes = ['Concepts:Add']
if validate_scopes(needed_write_scopes, myscopes_response.scopes):
  secrets_auth = user_or_secrets_auth
  secrets_stub = user_or_secrets_stub


filter_by = dict(
    query="LLM",
    # model_type_id="text-to-text",
)

API_INFO = list_all_models(filter_by=filter_by)

default_llms = get_default_models()

st.markdown(
    "<h1 style='text-align: center; color: black;'>LLM Battleground</h1>",
    unsafe_allow_html=True,
)

def reset_session():
  st.session_state['generated_completions'] = False
# st.markdown("Test out LLMs on a variety of tasks. See how they perform!")


def get_user():
  req = service_pb2.GetUserRequest(user_app_id=resources_pb2.UserAppIDSet(user_id="me"))
  response = user_or_secrets_stub.GetUser(req)
  if response.status.code != status_code_pb2.SUCCESS:
    show_error("GetUser", response)
  return response.user


if unauthorized:
  caller_id = "Anonymous"
else:
  user = get_user()
  caller_id = user.id


def create_prompt_model(model_id, prompt, position):
  if position not in ["PREFIX", "SUFFIX", "TEMPLATE"]:
    raise Exception("Position must be PREFIX or SUFFIX")

  # FIXME(zeiler): i think that if the user is logged in then postmodeloutputs or
  # postworkflowresults will fail because these models/workflows are not made publicly visible.
  # When the are anonymous then we fall back to the PAT provided in the secrets file.
  response = secrets_stub.PostModels(
      service_pb2.PostModelsRequest(
          user_app_id=userDataObject,
          models=[
              resources_pb2.Model(
                  id=model_id,
                  model_type_id="prompter",
              ),
          ],
      ))
  
  #response2= 

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
  post_model_versions_response = secrets_stub.PostModelVersions(req)
  if post_model_versions_response.status.code != status_code_pb2.SUCCESS:
    show_error("PostModelVersions", response)

  return post_model_versions_response.model

@st.cache_resource

#Used predict_by_bytes new SDK function.
def run_model(input_text, model):

  m = API_INFO[model]
  user_model_obj=Model(model_id=m["model_id"], user_id=m["user_id"], app_id=m["app_id"], pat=secrets_auth.pat)
  while True:
    try:
      response = user_model_obj.predict_by_bytes(bytes(
      input_text, 'utf-8'),
      "text")

    except Exception as e:
      st.error(f"Model predict error : {e} ")
      st.stop()

    break
  
  if DEBUG:
    st.json(json_format.MessageToDict(response, preserving_proto_field_name=True))

  return response


@st.cache_resource
def post_input(txt,id,concepts=[],metadata=None):
  """Posts input to the API and returns the response."""
  metadata_struct = Struct()
  metadata_struct.update(metadata)
  metadata = metadata_struct
  try:
    input_job_id = Inputs(
      logger_level="ERROR",
      user_id=userDataObject.user_id,
      app_id=userDataObject.app_id).upload_from_bytes(
        id,text_bytes=bytes(txt,'UTF-8'),
        labels=concepts,
        metadata=metadata)
                                                                                                             
  except Exception as e:
    st.error(f"post input error:{e}")
    #st.stop
  
  return input_job_id


def list_concepts():
  """Lists all concepts in the user's app."""
  response = secrets_stub.ListConcepts(
      service_pb2.ListConceptsRequest(user_app_id=userDataObject,))
  if response.status.code != status_code_pb2.SUCCESS:
    show_error("ListConcepts", response)
  return response.concepts


def post_concept(concept):
  """Posts a concept to the user's app."""
  response = secrets_stub.PostConcepts(
      service_pb2.PostConceptsRequest(
          user_app_id=userDataObject,
          concepts=[concept],
      ))
  if response.status.code != status_code_pb2.SUCCESS:
    show_error("PostConcepts", response)
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
  response = secrets_stub.PostAnnotationsSearches(req)
  # st.write(response)

  if response.status.code != status_code_pb2.SUCCESS:
    show_error("PostAnnotationsSearches", response)
  return response


@st.cache_resource
def get_input(input_id):
  """Searches for inputs in the user's app."""
  req = service_pb2.GetInputRequest(user_app_id=userDataObject, input_id=input_id)
  response = secrets_stub.GetInput(req)
  # st.write(response)

  if response.status.code != status_code_pb2.SUCCESS:
    show_error("GetInput", response)
  return response.input


def get_text(auth, url):
  """Download the raw text from the url"""
  try:
    h = {"Authorization": f"Key {auth.pat}"}
    response = requests.get(url, headers=h)
    response.encoding = response.apparent_encoding
  except Exception as e:
    print(f"Error: {e}")
    response = None
  return response.text if response else ""


# Check if prompt, completion and input are concepts in the user's app
app_concepts = list_concepts()
for concept in [PROMPT_CONCEPT,INPUT_CONCEPT, COMPLETION_CONCEPT]:
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
for concept in [PROMPT_CONCEPT,INPUT_CONCEPT, COMPLETION_CONCEPT]:
  if concept.id not in app_concept_ids:
    concepts_ready_bool = False

# Check if all required concepts are in the app
if not concepts_ready_bool:
  st.error("Need to add all the required concepts to the app before continuing.")
  st.stop()

input_search_response = search_inputs(concepts=[INPUT_CONCEPT], per_page=12)
completion_search_response = search_inputs(concepts=[COMPLETION_CONCEPT], per_page=12)

query_params = st.experimental_get_query_params()
inp = ""
if "inp" in query_params:
  input_id = query_params["inp"][0]
  res = get_input(input_id)
  inp = get_text(secrets_auth, res.data.text.url)

st.markdown(
    "<h2 style='text-align: center; color: black;'>Try many LLMs at once, see what works best and share</h2>",
    unsafe_allow_html=True,
)

model_names = sorted(API_INFO.keys())
models = st.multiselect("Select the LLMs you want to use:", model_names, default=default_llms, on_change=reset_session)

inp = st.text_area(
    " ",
    placeholder="Send a message to the LLMs",
    value=inp,
    help="Genenerate outputs from the LLMs using this input.", on_change=reset_session)


def render_card(container, input, caller_id, completions):
  c = "Completions"
  if caller_id is not None:
    c += f" (by: {caller_id})"
  container.markdown(
      f"<h1 style='text-align: center;font-size: 40px;color: #667085;'>{c}</h1>",
      unsafe_allow_html=True,
  )

  if input is not None:  # none for when using the text_input field.
    container.markdown(
        f"<h4 style='text-align: center;font-size: 20px;color: #667085;'>Input</h4>",
        unsafe_allow_html=True,
    )
    container.code(input, language=None)  # metric(label="Input", value=txt)

  container.markdown(
      f"<h4 style='text-align: center;font-size: 20px;color: #667085;'>Completions</h4>",
      unsafe_allow_html=True,
  )
  for d in completions:
    model_url = d['model'].split('/versions')[0]
    container.markdown(
        f"<a style='font-size: 12px;color: #667085;' href='{model_url}'>{model_url}</a>",
        unsafe_allow_html=True,
    )
    container.code(d['completion'], language=None)


generate_btn = st.button("Generate Completions")
if generate_btn:
  st.session_state['generated_completions'] = True

if st.session_state['generated_completions']:

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
    #input id     
    id = hashlib.md5(inp.encode("utf-8")).hexdigest() 
    inp_job_id = post_input(inp,id,concepts=["input"],metadata={"tags": ["input"],"caller": caller_id})
    # st.markdown(
    #     "<h1 style='text-align: center;font-size: 40px;color: #667085;'>Completions</h1>",
    #     unsafe_allow_html=True,
    # )

    cols = st.columns(3)
    h = ClarifaiUrlHelper(user_or_secrets_auth)
    link = h.clarifai_url(userDataObject.user_id, userDataObject.app_id,
                          "installed_module_versions", query_params["imv_id"][0])
    link = f"{link}?inp={id}"

    for mi, model in enumerate(models):
      col = cols[mi % len(cols)]
      container = col.container()
      prediction = run_model(inp, model)
      m = API_INFO[model]
      model_url = h.clarifai_url(m["user_id"], m["app_id"], "models", m["model_id"])
      model_url_with_version = h.clarifai_url(m["user_id"], m["app_id"], "models", m["model_id"],
                                              m["version_id"])
      # container.write(f"Completion from {model_url}:")

      completion = prediction.outputs[0].data.text.raw
      # container.write(completion)
      completion_job_id = post_input(
          completion,
          id=hashlib.md5(completion.encode("utf-8")).hexdigest(),
          concepts=['completion'],
          metadata={
              "input_id": id ,
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

    render_card(st, inp, None, completions)

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

    # share on twitter.
    components.html(f"""
            <a href="https://twitter.com/share?ref_src=twsrc%5Etfw" class="twitter-share-button"
            data-text="Check this cool @streamlit module built on @clarifai to compare @openai GPT-4 vs @AnthropicAI Claude 2 head to head, try it yourself!"
            data-url={link}
            data-show-count="false">
            data-size="Large"
            data-hashtags="streamlit,python,clarifai,llm"
            Tweet
            </a>
            <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
        """)

st.markdown(
    "Note: your messages and completions will be stored and shares publicly as recent messages")

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
        txt = get_text(secrets_auth, completion_hit.input.data.text.url)
        model_url = completion_hit.input.data.metadata.fields["model"].string_value
        completions.append({"completion": txt, "model": model_url})
    return completions

  previous_inputs = []

  cols = cycle(st.columns(3))
  for idx, input_hit in enumerate(input_search_response.hits):
    txt = get_text(secrets_auth, input_hit.input.data.text.url)
    previous_inputs.append({
        "input": txt,
    })
    # container = next(cols).container()
    container = st
    metadata = json_format.MessageToDict(input_hit.input.data.metadata)
    caller_id = metadata.get("caller", "zeiler")
    if caller_id == "":
      caller_id = "zeiler"

    # container.subheader(f"Input ({caller_id})", anchor=False)
    # container.code(txt, language=None)  # metric(label="Input", value=txt)

    # container.subheader("Completions:", anchor=False)

    completions = completions_for_input(input_hit.input.id)

    render_card(container, txt, caller_id, completions)

    if idx != len(input_search_response.hits) - 1:
      container.divider()

    # for tup in completions:
    #   txt, model_url = tup
    #   container.code(txt, language=None)
    #   ClarifaiStreamlitCSS.buttonlink(container, "Use Model", model_url)
    # container.write(completions)

cols = st.columns(3)
cols[0].markdown(
    "<a href='https://github.com/Clarifai/module-llm-battleground'><img style='height:24px;' src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png'/> See how this module was built </a>",
    unsafe_allow_html=True)
cols[1].markdown(
    "<a href='https://docs.clarifai.com/portal-guide/modules/create-install'>📚Learn how to build your own module </a>",
    unsafe_allow_html=True)
cols[2].markdown(
    "<a href='https://discord.com/invite/26upV8Y4Nd'><img src='https://www.clarifai.com/hubfs/discord-icon.svg'/> Request a feature</a>",
    unsafe_allow_html=True)

st.markdown(
    "<h3 style='text-align: center; color: black;'>Built on Clarifai with 💙 </h3>",
    unsafe_allow_html=True,
)
