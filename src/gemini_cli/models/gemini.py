"""
Gemini model integration for the CLI tool.
"""

import google.generativeai as genai
from google.generativeai import protos
from google.generativeai.types import FunctionDeclaration, Tool
import logging
import time
from rich.console import Console
from rich.panel import Panel
import questionary
import os

# Import exceptions for specific error handling if needed later
from google.api_core.exceptions import ResourceExhausted

from ..utils import count_tokens
from ..tools import get_tool, AVAILABLE_TOOLS

# Setup logging (basic config, consider moving to main.py)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
log = logging.getLogger(__name__)

MAX_AGENT_ITERATIONS = 10
FALLBACK_MODEL = "gemini-1.5-pro-latest"
CONTEXT_TRUNCATION_THRESHOLD_TOKENS = 800000 # Example token limit

def list_available_models(api_key):
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        gemini_models = []
        for model in models:
            # Filter for models supporting generateContent to avoid chat-only models if needed
            if 'generateContent' in model.supported_generation_methods:
                 model_info = { "name": model.name, "display_name": model.display_name, "description": model.description, "supported_generation_methods": model.supported_generation_methods }
                 gemini_models.append(model_info)
        return gemini_models
    except Exception as e:
        log.error(f"Error listing models: {str(e)}")
        return [{"error": str(e)}]


class GeminiModel:
    """Interface for Gemini models using native function calling agentic loop."""

    # --- MODIFIED: Added initial_context parameter --- 
    def __init__(self, api_key: str, console: Console, model_name: str ="gemini-2.5-pro-exp-03-25", initial_context: str | None = None):
        """Initialize the Gemini model interface."""
        self.api_key = api_key
        self.initial_model_name = model_name
        self.current_model_name = model_name
        self.console = console
        self.initial_context = initial_context or "Error: Initial directory context was not provided." # Store initial context
        genai.configure(api_key=api_key)
        # --- END MODIFIED ---

        self.generation_config = genai.types.GenerationConfig(temperature=0.4, top_p=0.95, top_k=40)
        self.safety_settings = { "HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE", "HATE": "BLOCK_MEDIUM_AND_ABOVE", "SEXUAL": "BLOCK_MEDIUM_AND_ABOVE", "DANGEROUS": "BLOCK_MEDIUM_AND_ABOVE" }
        
        # --- Tool Definition ---
        self.function_declarations = self._create_tool_definitions()
        self.gemini_tools = Tool(function_declarations=self.function_declarations) if self.function_declarations else None
        # ---

        # --- System Prompt (Native Functions & Planning) ---
        self.system_instruction = self._create_system_prompt()
        # ---

        # --- Initialize Persistent History --- 
        # --- MODIFIED: Use provided initial_context --- 
        self.chat_history = [
            # System instruction is now passed directly to the model instance
            # {'role': 'user', 'parts': [self.system_instruction]}, 
            # Model doesn't need to acknowledge readiness here, context is provided with first user prompt
            # {'role': 'model', 'parts': ["Okay, I'm ready. Provide the directory context and your request."]}
        ]
        log.info("Initialized persistent chat history (will add context with first user prompt).")
        # --- END MODIFIED ---
        # ---

        try:
            self._initialize_model_instance() # Creates self.model
            log.info("GeminiModel initialized successfully (Native Function Calling Agent Loop).")
        except Exception as e:
             log.error(f"Fatal error initializing Gemini model '{self.current_model_name}': {str(e)}", exc_info=True)
             raise Exception(f"Could not initialize Gemini model: {e}") from e

    def _initialize_model_instance(self):
        """Helper to create the GenerativeModel instance."""
        log.info(f"Initializing model instance: {self.current_model_name}")
        try:
            # Pass system instruction here, tools are passed during generate_content
            self.model = genai.GenerativeModel(
                model_name=self.current_model_name,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                system_instruction=self.system_instruction
            )
            log.info(f"Model instance '{self.current_model_name}' created successfully.")
        except Exception as init_err:
            log.error(f"Failed to create model instance for '{self.current_model_name}': {init_err}", exc_info=True)
            raise init_err

    def get_available_models(self):
        return list_available_models(self.api_key)

    # --- Native Function Calling Agent Loop ---
    def generate(self, prompt: str) -> str | None:
        logging.info(f"Agent Loop - Processing prompt: '{prompt[:100]}...' using model '{self.current_model_name}'")
        original_user_prompt = prompt
        if prompt.startswith('/'):
             command = prompt.split()[0].lower()
             # Handle commands like /compact here eventually
             if command in ['/exit', '/help']:
                 logging.info(f"Handled command: {command}")
                 return None # Or return specific help text

        # === Step 1: Prepare Initial User Turn (IF history is empty) ===
        # --- MODIFIED: Combine initial context ONLY for the very first user message --- 
        if not self.chat_history: # Check if history is empty (first turn)
            turn_input_prompt = f"{self.initial_context}\n\nUser request: {original_user_prompt}"
            log.info("First turn: Combining initial context with user prompt.")
            self.chat_history.append({'role': 'user', 'parts': [turn_input_prompt]})
        else:
            # For subsequent turns, just add the user's prompt directly
            log.info("Subsequent turn: Adding user prompt directly to history.")
            self.chat_history.append({'role': 'user', 'parts': [original_user_prompt]})
        # --- END MODIFIED ---
        
        # === START DEBUG LOGGING ===
        log.debug(f"Input added to chat_history (last item):\n---\n{self.chat_history[-1]}\n---")
        # === END DEBUG LOGGING ===
        self._manage_context_window() # Truncate *before* sending the request

        iteration_count = 0
        task_completed = False
        final_summary = None
        last_text_response = "No response generated." # Fallback text

        try:
            while iteration_count < MAX_AGENT_ITERATIONS:
                iteration_count += 1
                logging.info(f"Agent Loop Iteration {iteration_count}/{MAX_AGENT_ITERATIONS}")
                
                # === Call LLM with History and Tools ===
                llm_response = None
                try:
                    logging.info(f"Sending request to LLM ({self.current_model_name}). History length: {len(self.chat_history)} turns.")
                    # === ADD STATUS FOR LLM CALL ===
                    with self.console.status(f"[yellow]Assistant thinking ({self.current_model_name})...", spinner="dots"):
                        # Pass the available tools to the generate_content call
                        llm_response = self.model.generate_content(
                            self.chat_history,
                            generation_config=self.generation_config,
                            tools=[self.gemini_tools] if self.gemini_tools else None
                        )
                    # === END STATUS ===
                    
                    # === START DEBUG LOGGING ===
                    log.debug(f"RAW Gemini Response Object (Iter {iteration_count}): {llm_response}")
                    # === END DEBUG LOGGING ===
                    
                    # Extract the response part (candidate)
                    # Add checks for empty candidates or parts
                    if not llm_response.candidates:
                         log.error(f"LLM response had no candidates. Response: {llm_response}")
                         last_text_response = "(Agent received response with no candidates)"
                         task_completed = True; final_summary = last_text_response; break
                         
                    response_candidate = llm_response.candidates[0]
                    if not response_candidate.content or not response_candidate.content.parts:
                        log.error(f"LLM response candidate had no content or parts. Candidate: {response_candidate}")
                        last_text_response = "(Agent received response candidate with no content/parts)"
                        task_completed = True; final_summary = last_text_response; break

                    # --- REVISED LOOP LOGIC FOR MULTI-PART HANDLING --- 
                    function_call_part_to_execute = None
                    text_response_buffer = ""
                    processed_function_call_in_turn = False # Flag to ensure only one function call is processed per turn

                    # Iterate through all parts in the response
                    for part in response_candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call and not processed_function_call_in_turn:
                            function_call = part.function_call
                            tool_name = function_call.name
                            tool_args = dict(function_call.args) if function_call.args else {}
                            log.info(f"LLM requested Function Call: {tool_name} with args: {tool_args}")

                            # Add the function *call* part to history immediately
                            self.chat_history.append({'role': 'model', 'parts': [part]})
                            self._manage_context_window()
                            
                            # Store details for execution after processing all parts
                            function_call_part_to_execute = part 
                            processed_function_call_in_turn = True # Mark that we found and will process a function call
                            # Don't break here yet, process other parts (like text) first for history/logging

                        elif hasattr(part, 'text') and part.text:
                            llm_text = part.text
                            log.info(f"LLM returned text part (Iter {iteration_count}): {llm_text[:100]}...")
                            text_response_buffer += llm_text + "\n" # Append text parts
                            # Add the text response part to history
                            self.chat_history.append({'role': 'model', 'parts': [part]})
                            self._manage_context_window()
                            
                        else:
                            log.warning(f"LLM returned unexpected response part (Iter {iteration_count}): {part}")
                            # Add it to history anyway?
                            self.chat_history.append({'role': 'model', 'parts': [part]})
                            self._manage_context_window()

                    # --- Now, decide action based on processed parts --- 
                    if function_call_part_to_execute:
                        # === Execute the Tool === (Using stored details)
                        function_call = function_call_part_to_execute.function_call # Get the stored call
                        tool_name = function_call.name
                        tool_args = dict(function_call.args) if function_call.args else {}
                        
                        tool_result = ""
                        tool_error = False
                        user_rejected = False # Flag for user rejection
                        
                        # --- HUMAN IN THE LOOP CONFIRMATION ---
                        # --- MODIFIED: Enhanced Confirmation Logic ---
                        if tool_name in ["edit", "create_file", "formatter", "create_directory"]: # Added formatter & create_directory
                            # Determine the primary path argument based on the tool
                            if tool_name == "create_directory":
                                file_path = tool_args.get("dir_path", "(unknown directory)")
                            else: # edit, create_file, formatter
                                file_path = tool_args.get("path") or tool_args.get("file_path", "(unknown file)")

                            content = tool_args.get("content") # Get content, might be None
                            old_string = tool_args.get("old_string") # Get old_string
                            new_string = tool_args.get("new_string") # Get new_string
                            
                            action_description = f"Apply changes using [bold cyan]{tool_name}[/bold cyan] to [bold magenta]{file_path}[/bold magenta]?"
                            panel_content = ""
                            
                            if tool_name == "formatter":
                                panel_content = "[yellow]Action:[/yellow] Format file content."
                            elif tool_name == "create_directory":
                                panel_content = "[yellow]Action:[/yellow] Create new directory."
                            elif content is not None: # Case 1: Full content provided (create or overwrite)
                                action_verb = "Create/Overwrite" # Simplified, assume overwrite is possible
                                if os.path.exists(file_path) and os.path.isdir(file_path):
                                     action_verb = "[bold red]Error: Path is a directory[/bold red]" # Prevent overwriting dir
                                elif not os.path.exists(os.path.dirname(os.path.abspath(file_path))):
                                     action_verb = "Create file (and parent directories)"
                                elif not os.path.exists(file_path):
                                     action_verb = "Create file"
                                     
                                panel_content = f"[yellow]Action:[/yellow] {action_verb}.\n"
                                # Prepare content preview (limit length?)
                                preview_lines = content.splitlines()
                                max_preview_lines = 25 # Limit preview
                                if len(preview_lines) > max_preview_lines:
                                    content_preview = "\n".join(preview_lines[:max_preview_lines]) + f"\n... ({len(preview_lines) - max_preview_lines} more lines)"
                                else:
                                    content_preview = content
                                panel_content += f"\n[bold]Content Preview:[/bold]\n---\n{content_preview}\n---"
                                
                            elif old_string is not None and new_string is not None: # Case 2: Replacement
                                panel_content = "[yellow]Action:[/yellow] Replace first occurrence.\n"
                                max_snippet = 70 # Max chars to show for old/new strings
                                old_snippet = old_string[:max_snippet] + ('...' if len(old_string) > max_snippet else '')
                                new_snippet = new_string[:max_snippet] + ('...' if len(new_string) > max_snippet else '')
                                # --- FIXED: Removed trailing quote --- 
                                panel_content += f"\n[bold red]- Remove:[/bold red]\n---\n{old_snippet}\n---"
                                # --- END FIXED ---
                                panel_content += f"\n[bold green]+ Add:[/bold green]\n---\n{new_snippet}\n---"
                            else: # Case 3: Other/Unknown edit args or empty file creation
                                 panel_content = "[yellow]Action:[/yellow] Create empty file or unknown edit operation."

                            # Use Rich Panel for better presentation
                            self.console.print(Panel(
                                panel_content, 
                                title="Confirm File/Directory Modification", # Updated title
                                border_style="red",
                                expand=False,
                                subtitle=f"Tool: {tool_name}"
                            ))
                            
                            # Use questionary for confirmation
                            confirmed = questionary.confirm(
                                action_description, # Use the clearer description
                                default=False, # Default to No
                                auto_enter=False # Require Enter key press
                            ).ask()
                        # --- END MODIFIED ---
                            
                            # Handle case where user might Ctrl+C during prompt
                            if confirmed is None: 
                                log.warning("User cancelled confirmation prompt.")
                                tool_result = f"User cancelled confirmation for {tool_name} on {file_path}."
                                user_rejected = True
                            elif not confirmed: # User explicitly selected No
                                log.warning(f"User rejected proposed action: {tool_name} on {file_path}")
                                tool_result = f"User rejected the proposed {tool_name} operation on {file_path}."
                                user_rejected = True # Set flag to skip execution
                            else: # User selected Yes
                                log.info(f"User confirmed action: {tool_name} on {file_path}")
                        # --- END CONFIRMATION ---

                        # Only execute if not rejected by user
                        if not user_rejected:
                            status_msg = f"Executing {tool_name}"
                            if tool_args: status_msg += f" ({', '.join([f'{k}={str(v)[:30]}...' if len(str(v))>30 else f'{k}={v}' for k,v in tool_args.items()])})"
                            
                            with self.console.status(f"[yellow]{status_msg}...", spinner="dots"):
                                try:
                                    # --- MODIFIED: Handle SummarizeCodeTool instantiation --- 
                                    if tool_name == "summarize_code":
                                        tool_instance = AVAILABLE_TOOLS.get(tool_name)
                                        if tool_instance:
                                            # Pass the model instance ONLY to SummarizeCodeTool
                                            tool_instance = tool_instance(model_instance=self.model)
                                        else: 
                                            tool_instance = None # Should not happen if AVAILABLE_TOOLS is correct
                                    else:
                                        # Standard instantiation for other tools
                                        tool_instance = get_tool(tool_name)
                                    # --- END MODIFIED ---
                                    
                                    if tool_instance:
                                        log.debug(f"Executing tool '{tool_name}' with arguments: {tool_args}")
                                        tool_result = tool_instance.execute(**tool_args)
                                        log.info(f"Tool '{tool_name}' executed. Result length: {len(str(tool_result)) if tool_result else 0}")
                                        log.debug(f"Tool '{tool_name}' result: {str(tool_result)[:500]}...")
                                        # --- MODIFIED: Check for errors in tool_result string --- 
                                        if isinstance(tool_result, str) and tool_result.strip().startswith("Error:"):
                                            log.error(f"Tool '{tool_name}' reported an error: {tool_result}")
                                            tool_error = True
                                        # --- END MODIFIED ---
                                    else:
                                        log.error(f"Tool '{tool_name}' not found.")
                                        tool_result = f"Error: Tool '{tool_name}' is not available."
                                        tool_error = True
                                except Exception as tool_exec_error:
                                    log.error(f"Error executing tool '{tool_name}' with args {tool_args}: {tool_exec_error}", exc_info=True)
                                    tool_result = f"Error executing tool {tool_name}: {str(tool_exec_error)}"
                                    tool_error = True
                                
                                # --- Print Executed/Error INSIDE the status block --- 
                                # --- MODIFIED: Enhanced Error Display --- 
                                if tool_error:
                                    # Use Rich Panel for errors
                                    self.console.print(Panel(
                                        f"[bold]Tool:[/bold] {tool_name}\n[bold]Arguments:[/bold] {tool_args}\n\n[white]{str(tool_result)}[/white]",
                                        title="Tool Execution Error",
                                        border_style="bold red",
                                        expand=False
                                    ))
                                else:
                                    self.console.print(f"[dim] -> Executed {tool_name}[/dim]") 
                                # --- END MODIFIED ---
                            # --- End Status Block ---
                                
                        # === Check for Task Completion Signal via Tool Call ===
                        if tool_name == "task_complete":
                            log.info("Task completion signaled by 'task_complete' function call.")
                            task_completed = True
                            final_summary = tool_result # The result of task_complete IS the summary
                            # We break *after* adding the function response below
                        
                        # === Add Function Response to History ===
                        # Create the FunctionResponse proto
                        function_response_proto = protos.FunctionResponse(
                            name=tool_name,
                            response={"result": tool_result} # API expects dict
                        )
                        # Wrap it in a Part proto
                        response_part_proto = protos.Part(function_response=function_response_proto)
                        
                        # Append to history
                        self.chat_history.append({'role': 'user', # Function response acts as a 'user' turn providing data
                                              'parts': [response_part_proto]})
                        self._manage_context_window()
                        
                        if task_completed: 
                            break # Exit loop NOW that task_complete result is in history
                        # --- MODIFIED: Add continue even if tool errored, let LLM react --- 
                        # else:
                        #     continue # IMPORTANT: Continue loop to let LLM react to function result
                        continue # Always continue after a function call/response cycle
                        # --- END MODIFIED ---
                            
                    elif text_response_buffer: 
                        # === Only Text Returned ===
                        log.info("LLM returned only text response(s). Assuming task completion or explanation provided.")
                        last_text_response = text_response_buffer.strip()
                        task_completed = True # Treat text response as completion
                        final_summary = last_text_response # Use the text as the summary
                        break # Exit the loop
                    
                    else:
                        # === No actionable parts found ===
                        log.warning("LLM response contained no actionable parts (text or function call).")
                        last_text_response = "(Agent received response with no actionable parts)"
                        task_completed = True # Treat as completion to avoid loop errors
                        final_summary = last_text_response
                        break # Exit loop

                except ResourceExhausted as quota_error:
                    log.warning(f"Quota exceeded for model '{self.current_model_name}': {quota_error}")
                    # Check if we are already using the fallback
                    if self.current_model_name == self.FALLBACK_MODEL:
                        log.error("Quota exceeded even for the fallback model. Cannot proceed.")
                        self.console.print(f"[bold red]API quota exceeded for primary and fallback models. Please check your plan/billing.[/bold red]")
                        # Clean history before returning
                        if self.chat_history[-1]['role'] == 'user': self.chat_history.pop()
                        return f"Error: API quota exceeded for primary and fallback models."
                    else:
                        log.info(f"Switching to fallback model: {self.FALLBACK_MODEL}")
                        self.console.print(f"[bold yellow]Quota limit reached for {self.current_model_name}. Switching to fallback model ({self.FALLBACK_MODEL})...[/bold yellow]")
                        self.current_model_name = self.FALLBACK_MODEL
                        try:
                            self._initialize_model_instance() # Recreate model instance with fallback name
                            log.info(f"Successfully switched to and initialized fallback model: {self.current_model_name}")
                            # Important: Clear the last model response (which caused the error) before retrying
                            if self.chat_history[-1]['role'] == 'model': 
                               last_part = self.chat_history[-1]['parts'][0]
                               # Only pop if it was a failed function call attempt or empty text response leading to error
                               if hasattr(last_part, 'function_call') or not hasattr(last_part, 'text') or not last_part.text:
                                   self.chat_history.pop() 
                                   log.debug("Removed last model part before retrying with fallback.")
                            continue # Retry the current loop iteration with the new model
                        except Exception as fallback_init_error:
                            log.error(f"Failed to initialize fallback model '{self.FALLBACK_MODEL}': {fallback_init_error}", exc_info=True)
                            self.console.print(f"[bold red]Error switching to fallback model: {fallback_init_error}[/bold red]")
                            if self.chat_history[-1]['role'] == 'user': self.chat_history.pop()
                            return f"Error: Failed to initialize fallback model after quota error."

                except Exception as generation_error:
                     # This handles other errors during the generate_content call or loop logic
                     log.error(f"Error during Agent Loop: {generation_error}", exc_info=True)
                     # Clean history
                     if self.chat_history[-1]['role'] == 'user': self.chat_history.pop()
                     return f"Error during agent processing: {generation_error}"

            # === End Agent Loop ===

            # === Handle Final Output ===
            if task_completed and final_summary:
                 log.info("Agent loop finished. Returning final summary.")
                 # Cleanup internal tags if needed (using a hypothetical method)
                 # cleaned_summary = self._cleanup_internal_tags(final_summary) 
                 return final_summary.strip() # Return the summary from task_complete or final text
            elif iteration_count >= MAX_AGENT_ITERATIONS:
                 log.warning(f"Agent loop terminated after reaching max iterations ({MAX_AGENT_ITERATIONS}).")
                 # Try to get the last *text* response the model generated, even if it wanted to call a function after
                 last_model_response_text = self._find_last_model_text(self.chat_history)
                 timeout_message = f"(Task exceeded max iterations ({MAX_AGENT_ITERATIONS}). Last text from model was: {last_model_response_text})"
                 return timeout_message.strip()
            else:
                 # This case should be less likely now
                 log.error("Agent loop exited unexpectedly.")
                 last_model_response_text = self._find_last_model_text(self.chat_history)
                 return f"(Agent loop finished unexpectedly. Last model text: {last_model_response_text})"

        except Exception as e:
             log.error(f"Error during Agent Loop: {str(e)}", exc_info=True)
             # --- MODIFIED: Clearer error message --- 
             return f"An unexpected error occurred during the agent process: {str(e)}"
             # --- END MODIFIED ---

    # --- Context Management (Consider Token Counting) ---
    def _manage_context_window(self):
        """Basic context window management based on turn count."""
        # Placeholder - Enhance with token counting
        MAX_HISTORY_TURNS = 20 # Keep ~N pairs of user/model turns + initial setup + tool calls/responses
        # Each full LLM round (request + function_call + function_response) adds 3 items
        if len(self.chat_history) > (MAX_HISTORY_TURNS * 3 + 2): 
             log.warning(f"Chat history length ({len(self.chat_history)}) exceeded threshold. Truncating.")
             # Keep system prompt (idx 0), initial model ack (idx 1)
             keep_count = MAX_HISTORY_TURNS * 3 # Keep N rounds
             keep_from_index = len(self.chat_history) - keep_count
             self.chat_history = self.chat_history[:2] + self.chat_history[keep_from_index:]
             log.info(f"History truncated to {len(self.chat_history)} items.")

    # --- Tool Definition Helper ---
    def _create_tool_definitions(self) -> list[FunctionDeclaration] | None:
        """Dynamically create FunctionDeclarations from AVAILABLE_TOOLS."""
        declarations = []
        for tool_name, tool_instance in AVAILABLE_TOOLS.items():
            if hasattr(tool_instance, 'get_function_declaration'):
                declaration = tool_instance.get_function_declaration()
                if declaration:
                    declarations.append(declaration)
                    log.debug(f"Generated FunctionDeclaration for tool: {tool_name}")
                else:
                    log.warning(f"Tool {tool_name} has 'get_function_declaration' but it returned None.")
            else:
                # Fallback or skip tools without the method? For now, log warning.
                log.warning(f"Tool {tool_name} does not have a 'get_function_declaration' method. Skipping.")
        
        log.info(f"Created {len(declarations)} function declarations for native tool use.")
        return declarations if declarations else None

    # --- System Prompt Helper ---
    def _create_system_prompt(self) -> str:
        """Creates the system prompt, emphasizing native functions and planning."""
        # Use docstrings from tools if possible for descriptions
        tool_descriptions = []
        if self.function_declarations:
            for func_decl in self.function_declarations:
                 # Simple representation: name(args) - description
                 # Ensure parameters exist before trying to access properties
                 args_str = ""
                 if func_decl.parameters and func_decl.parameters.properties:
                      args_list = []
                      required_args = func_decl.parameters.required or []
                      for prop, details in func_decl.parameters.properties.items():
                            # Access attributes directly from the Schema object
                            prop_type = details.type if hasattr(details, 'type') else 'UNKNOWN' 
                            prop_desc = details.description if hasattr(details, 'description') else ''
                            
                            suffix = "" if prop in required_args else "?" # Indicate optional args
                            
                            # Include parameter description in the string for clarity in the system prompt
                            args_list.append(f"{prop}: {prop_type}{suffix} # {prop_desc}") 
                            
                      args_str = ", ".join(args_list)
                 
                 desc = func_decl.description or "(No description provided)" # Overall func desc
                 tool_descriptions.append(f"- `{func_decl.name}({args_str})`: {desc}")
        else:
             tool_descriptions.append(" - (No tools available with function declarations)")

        tool_list_str = "\n".join(tool_descriptions)

        # --- MODIFIED: Prompt v13.3 - Enhanced Clarity & Interaction ---
        return f"""You are Gemini Code, an AI coding assistant running in a CLI environment.
Your primary goal is to assist the user with coding tasks by understanding their request, planning steps, and utilizing available tools via **native function calls**. Strive for clarity, accuracy, and proactive assistance.

Available Tools (Use ONLY these via function calls):
{tool_list_str}

Core Workflow:
1.  **Analyze & Plan:** Carefully read the user's request and the provided directory context (`ls` output).
    *   **Ask Clarifying Questions:** If the request is ambiguous or lacks detail, ask specific questions before proceeding (e.g., "Which file should I modify?", "What should the function return?").
    *   **Outline Plan:** For non-trivial tasks involving multiple steps, respond with a brief, numbered plan outlining the tools you intend to use (e.g., 1. `view` file.py, 2. `edit` file.py, 3. `test_runner`).
    *   **State Intent:** Clearly state when you are about to perform an action, especially filesystem modifications (e.g., "Okay, I will now use the `edit` tool to add the function to `utils.py`. This requires confirmation.").
2.  **Execute Step:** Make the **single function call** needed for the current step in your plan.
    *   **Confirmation Required:** Remember that `edit`, `create_directory`, `formatter`, and potentially `bash` (depending on the command) require user confirmation. The system handles this prompt.
3.  **Observe & Adapt:** Analyze the result of the function call (or user rejection).
    *   If the tool succeeded, proceed to the next step in your plan.
    *   If the tool failed or the user rejected the action, explain the issue and suggest an alternative or ask for guidance.
4.  **Iterate:** Repeat steps 2 and 3 until the user's request is fully addressed.
5.  **Complete Task:** Once the *entire* task is finished successfully, **you MUST call the `task_complete` function**.
    *   Provide a concise, accurate `summary` in Markdown, detailing what was done (e.g., "Created `new_script.py` and added a basic function. You can run it with `python new_script.py`.").
    *   If code was changed, the summary **MUST** include specific commands to run or test the result (use Markdown code blocks).

Key Interaction Principles:
*   **Native Functions Only:** Interact with tools *exclusively* through function calls. Do NOT write tool calls as text (e.g., `cli_tools.ls(...)`).
*   **One Call Per Turn:** Execute only one function call per response. Wait for the result before the next call.
*   **Full Initial Context:** When asked generally about the workspace ("what's here?", "list files"), your *first* response must list or summarize **ALL** files/directories from the initial `ls` context. Do not filter or make assumptions. Use `tree` if hierarchy is important.
*   **Accurate Reporting:** When listing files/directories later, be comprehensive and accurate based on `ls` or `tree` output. Include config files, docs, etc.
*   **Explanations & Instructions:**
    *   If asked *how* to do something or for an explanation, provide it directly in a text response using clear Markdown.
    *   **Proactive Command Execution:** After explaining steps culminating in a command (e.g., `python run.py`, `pytest`, `git status | cat`), **ask the user** if they want you to run it using the appropriate tool (`bash`, `test_runner`). Example: "I've explained how to run the tests. Would you like me to execute `pytest` using the `test_runner` tool?" (Append `| cat` to `bash` commands that might page output).
    *   Do *not* call `task_complete` just for providing information.
*   **Precise Edits:** Prefer `view` with offset/limit before `edit`. Use `old_string`/`new_string` for targeted replacements. Use `content` mainly for new files or full overwrites.
*   **Error Handling:** If a tool fails, report the error clearly and suggest a fix or alternative approach.

The user's first message contains initial directory context and their request. Begin by analyzing this information.
"""
        # --- END MODIFIED ---

    # --- Text Extraction Helper (if needed for final output) ---
    def _extract_text_from_response(self, response) -> str | None:
         """Safely extracts text from a Gemini response object."""
         try:
             if response and response.candidates:
                 # Handle potential multi-part responses if ever needed, for now assume text is in the first part
                 if response.candidates[0].content and response.candidates[0].content.parts:
                     text_parts = [part.text for part in response.candidates[0].content.parts if hasattr(part, 'text')]
                     return "\n".join(text_parts).strip() if text_parts else None
             return None
         except (AttributeError, IndexError) as e:
             log.warning(f"Could not extract text from response: {e} - Response: {response}")
             return None
             
    # --- Find Last Text Helper ---
    def _find_last_model_text(self, history: list) -> str:
        """Finds the last text part sent by the model in the history."""
        for i in range(len(history) - 1, -1, -1):
            if history[i]['role'] == 'model':
                try:
                     # Check if parts exists and has content
                     if history[i]['parts'] and hasattr(history[i]['parts'][0], 'text'):
                           return history[i]['parts'][0].text.strip()
                except (AttributeError, IndexError):
                     continue # Ignore malformed history entries
        return "(No previous text response found)"