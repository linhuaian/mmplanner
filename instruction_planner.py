from openai import OpenAI

class InstructionPlanner:
    def __init__(self, api_key):

        self.client = OpenAI(
            api_key=api_key,
            base_url='https://compass.llm.shopee.io/compass-api/v1',
        )

    def generate_text_plan(self, instruction, min_steps=5, max_steps=10):
        prompt = (
            f"Break down the instruction \"{instruction}\" into a step-by-step plan. "
            f"Generate between {min_steps} and {max_steps} clear, concise steps, generate only the step plan without other words."
        )
        chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model="gpt-4o",
                    temperature=0.01,
                    extra_headers={
                        "Provider": "OpenAI"
                    }
                )
        content = chat_completion.choices[0].message.content
        steps = [line.strip() for line in content.strip().split('\n') if line.strip()]
        return steps

if __name__ == "__main__":
    ip = InstructionPlanner("56d5b8ebf4d5ec1fd1731af1cfa971be9cf711c0a0a366b01b42971937e6bf69")
    plan = ip.generate_text_plan("How to cook a fried egg?")
    print(plan)

