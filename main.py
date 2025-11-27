from instruction_planner import InstructionPlanner


if __name__ == "__main__":
    planner = InstructionPlanner(api_key="YOUR_OPENAI_API_KEY")
    text_plan = planner.generate_text_plan("how to cook an egg")
    print(text_plan)
