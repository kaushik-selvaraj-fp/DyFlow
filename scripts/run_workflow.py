import sys
import os
import google.auth

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dyflow import WorkflowExecutor, ModelService
import vertexai

_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# Initialize Vertex AI
try:
    vertexai.init(project=project_id, location="global")
    print(f"✓ Extractor Initialized with: {project_id}")
except Exception as e:
    print(f"Error initializing Vertex AI: {e}")
    exit(1)


def main():
    """
    Run DyFlow on a sample problem.
    """
    # Define the problem to solve
    problem_description = """
    Solve the following math problem:

    A farmer has 12 chickens and 8 rabbits on his farm.
    Each chicken has 2 legs and each rabbit has 4 legs.
    How many legs do all the animals have in total?

    Please provide a step-by-step solution.
    """

    problem_description = """
    Gold Investment Dilemma: You have exactly ₹1 lakh to spend on gold jewelry from three options with specific constraints. 
    A necklace must be at least 32 grams with no upper limit—you can go heavier if budget allows, and you'll wear one piece at a time. 
    Bangles range from 32-40 grams, offering flexibility to add more weight within that band. Rings present a unique challenge: each ring must weigh 2-8 grams, 
    but you need to buy two identical rings (one for each ear—likely earrings, not finger rings), meaning your actual purchase is 4-16 grams total in a matching pair. 
    Here's where it gets tricky: the necklace offers unlimited scalability and maximum neck-level visibility but carries 10-15% making charges on potentially the heaviest weight. 
    Bangles give you wrist-focused attention, rock-solid durability, and the lowest making charges (6-10%) across a fixed 32-40g range. 
    The ring pair provides the least total gold weight (4-16g) but requires double-item precision with the highest making charges (15-25% each) and faces durability concerns. 
    At current 22K gold rates, your money must cover base gold cost + making charges + 18% GST. 
    The real puzzle: should you maximize weight with an unrestricted necklace, optimize value with mid-range bangles, or split your budget across two delicate earrings—and which strategy delivers maximum social visibility per rupee while staying within your fixed budget after all charges?
    """

    # Create Designer and Executor services
    # Designer: Plans the problem-solving strategy
    # Executor: Executes the planned steps
    # designer_service = ModelService(model="gpt-4.1")
    # executor_service = ModelService(model="phi-4")

    designer_service = ModelService(model="gemini-2.5-pro")
    executor_service = ModelService(model="gemini-2.5-pro")

    # Alternative: Use local models
    # designer_service = ModelService.local()
    # executor_service = ModelService(model="phi-4")

    # Create workflow executor
    executor = WorkflowExecutor(
        problem_description=problem_description,
        designer_service=designer_service,
        executor_service=executor_service,
    )

    # Execute the adaptive workflow
    print("Starting DyFlow execution...")
    print("-" * 60)
    final_answer = executor.execute()

    # Display results
    print("\n" + "=" * 60)
    print("WORKFLOW EXECUTION COMPLETE")
    print("=" * 60)

    print("\n=== Final Answer ===")
    print(final_answer)

    print("\n=== Workflow Summary ===")
    print(executor.state.get_state_summary_for_designer())


if __name__ == "__main__":
    main()
