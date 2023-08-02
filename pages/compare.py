import hashlib
import os
import time
import uuid
from itertools import cycle
from typing import Dict, Iterator, List

import diff_viewer
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import V2Stub, create_stub
from clarifai.listing.lister import ClarifaiResourceLister
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format
from google.protobuf.json_format import MessageToDict

st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)


def local_css(file_name):
  with open(file_name) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)


# Note(zeiler): we need to store a special PAT to post inputs.
def load_pat():
  home = os.environ['HOME']
  if not os.path.exists(home + '/.clarifai_pat'):
    pat = st.text_input("Enter a Clarifai Personal Access Token that can write to this app")
    if pat:
      with open(home + '/.clarifai_pat', 'w') as f:
        f.write(pat)
    else:
      st.stop()

  with open(home + '/.clarifai_pat') as f:
    return f.read()
  
def show_error(request_name, response):
  st.error("There was an error with your request to "+request_name)
  st.json(json_format.MessageToJson(response, preserving_proto_field_name=True))
  raise Exception("There was an error with your request to "+request_name+" "+response.status.description)
  st.stop()

def models_generator(stub: V2Stub,
                     page_size: int = 64,
                     filter_by: dict = {},) -> Iterator[resources_pb2.Model]:
  """
    Iterator for all the community models based on specified conditions.

    Args:
      stub: client stub.
      page_size: the pagination size to use while iterating.
      filter_by: a dictionary of filters to apply to the list of models.

    Returns:
      models: a list of Model protos for all the community models.
    """
  userDataObject = resources_pb2.UserAppIDSet()
  model_success_status = {status_code_pb2.SUCCESS}

  page = 1
  while True:
    response = stub.ListModels(
        service_pb2.ListModelsRequest(user_app_id=userDataObject, page=page, per_page=page_size, **filter_by),)

    if response.status.code in model_success_status:
      show_error("ListModels", response)
    if len(response.models) == 0:
      break
    for item in response.models:
      yield item
    page += 1

def list_models(stub: V2Stub,
                filter_by: dict = {},)-> Dict[str, Dict[str, str]]:
    """
      Lists all the community models based on specified conditions.

      Args:
        stub: client stub.
        filter_by: a dictionary of filters to apply to the list of models.

      Returns:
        API_INFO: dictionary of models information.
    """
    API_INFO = {}
    for model_proto in models_generator(stub=stub, filter_by=filter_by):
        model_dict = MessageToDict(model_proto)
        try:
            API_INFO[f"{model_dict['id']}: {model_dict['userId']}"] = dict(user_id=model_dict["userId"],
                                                                            app_id=model_dict["appId"], model_id=model_dict["id"],
                                                                            version_id=model_dict["modelVersion"]["id"])
        except IndexError:
            pass
    return API_INFO

pat = load_pat()

local_css("./style.css")

DEBUG = False
completions = []

PROMPT_CONCEPT = resources_pb2.Concept(id="prompt", value=1.0)
INPUT_CONCEPT = resources_pb2.Concept(id="input", value=1.0)
COMPLETION_CONCEPT = resources_pb2.Concept(id="completion", value=1.0)

# This must be within the display() function.

# We need to use this post_input
input_auth = ClarifaiAuthHelper.from_streamlit(st)
input_auth._pat = pat
input_stub = create_stub(input_auth)

module_query_params = st.experimental_get_query_params()
# Check if the user is logged in. If not, use internal PAT.
if module_query_params.get("pat", "") == "" and module_query_params.get("token", "") == "":
  module_query_params["pat"] = [pat]
  auth = ClarifaiAuthHelper.from_streamlit_query_params(module_query_params)
  unauthorized = True
else:
  auth = ClarifaiAuthHelper.from_streamlit(st)
  unauthorized = False

stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()
lister = ClarifaiResourceLister(stub, auth.user_id, auth.app_id, page_size=16)

filter_by = dict(
    query="LLM",
    model_type_id="text-to-text",
)
API_INFO = list_models(stub, filter_by=filter_by)

st.markdown(
    "<h1 style='text-align: center; color: black;'>LLM Battleground</h1>",
    unsafe_allow_html=True,
)

# st.markdown("Test out LLMs on a variety of tasks. See how they perform!")


def get_user():
  req = service_pb2.GetUserRequest(user_app_id=resources_pb2.UserAppIDSet(user_id="me"))
  response = stub.GetUser(req)
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
    show_error("PostModels", response)

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
    show_error("PostModelVersions", response)

  return post_model_versions_response.model


def delete_model(model):
  response = stub.DeleteModels(
      service_pb2.DeleteModelsRequest(
          user_app_id=userDataObject,
          ids=[model.id],
      ))
  if response.status.code != status_code_pb2.SUCCESS:
    show_error("DeleteModels", response)


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
    show_error("PostWorkflows", response)

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
    show_error("DeleteWorkflows", response)
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
    show_error("PostWorkflowResults", response)

  if DEBUG:
    st.json(json_format.MessageToDict(response, preserving_proto_field_name=True))

  return response


@st.cache_resource
def run_model(input_text, model):

  m = API_INFO[model]
  start_time = time.time()
  while True:

    response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=resources_pb2.UserAppIDSet(user_id=m['user_id'], app_id=m['app_id']),
            model_id=m['model_id'],
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(text=resources_pb2.Text(raw=input_text,),),),
            ],
        ))

    if response.outputs[0].status.code == status_code_pb2.MODEL_DEPLOYING and time.time() - start_time < 60*10:
      st.info(f"{model.split(':')[0]} model is still deploying, please wait...")
      time.sleep(5)
      continue

    if response.status.code != status_code_pb2.SUCCESS:
      show_error("PostModelOutputs", response)
    else:
      break

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
  response = input_stub.PostInputs(req)
  if response.status.code != status_code_pb2.SUCCESS:
    if len(response.inputs) and response.inputs[0].status.details.find("duplicate ID") != -1:
      # If the input already exists, just return the input
      return req.inputs[0]
    show_error("PostInputs", response)
  return response.inputs[0]


def list_concepts():
  """Lists all concepts in the user's app."""
  response = input_stub.ListConcepts(service_pb2.ListConceptsRequest(user_app_id=userDataObject,))
  if response.status.code != status_code_pb2.SUCCESS:
    show_error("ListConcepts", response)
  return response.concepts


def post_concept(concept):
  """Posts a concept to the user's app."""
  response = input_stub.PostConcepts(
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
  response = input_stub.PostAnnotationsSearches(req)
  # st.write(response)

  if response.status.code != status_code_pb2.SUCCESS:
    show_error("PostAnnotationsSearches", response)
  return response


@st.cache_resource
def get_input(input_id):
  """Searches for inputs in the user's app."""
  req = service_pb2.GetInputRequest(user_app_id=userDataObject, input_id=input_id)
  response = input_stub.GetInput(req)
  # st.write(response)

  if response.status.code != status_code_pb2.SUCCESS:
    show_error("GetInput", response)
  return response.input


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

query_params = st.experimental_get_query_params()
inp = ""
if "inp" in query_params:
  input_id = query_params["inp"][0]
  res = get_input(input_id)
  inp = get_text(res.data.text.url)

st.markdown(
    "<h2 style='text-align: center; color: black;'>Try many LLMs at once, see what works best and share</h2>",
    unsafe_allow_html=True,
)

model_names = list(API_INFO.keys())
models = st.multiselect(
    "Select the LLMs you want to use:", model_names, default=['GPT-4: openai', 'claude-v2: anthropic'])

inp = st.text_area(
    " ",
    placeholder="Send a message to the LLMs",
    value=inp,
    help="Genenerate outputs from the LLMs using this input.")


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

if st.button("Generate Completions"):

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

    # st.markdown(
    #     "<h1 style='text-align: center;font-size: 40px;color: #667085;'>Completions</h1>",
    #     unsafe_allow_html=True,
    # )

    cols = st.columns(3)
    h = ClarifaiUrlHelper(auth)
    link = h.clarifai_url(userDataObject.user_id, userDataObject.app_id, "installed_module_versions",
                          query_params["imv_id"][0])
    link = f"{link}?inp={inp_input.id}"

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
        txt = get_text(completion_hit.input.data.text.url)
        model_url = completion_hit.input.data.metadata.fields["model"].string_value
        completions.append({"completion": txt, "model": model_url})
    return completions

  previous_inputs = []

  cols = cycle(st.columns(3))
  for idx, input_hit in enumerate(input_search_response.hits):
    txt = get_text(input_hit.input.data.text.url)
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
    "<a href='https://join.slack.com/t/clarifaicommunity/shared_invite/zt-1jehqesme-l60djcd3c_4a1eCV~uPUjQ'><img src='https://www.clarifai.com/hubfs/slack-icon.svg'/> Request a feature</a>",
    unsafe_allow_html=True)

st.markdown(
    "<h3 style='text-align: center; color: black;'>Built on Clarifai with 💙 </h3>",
    unsafe_allow_html=True,
)
