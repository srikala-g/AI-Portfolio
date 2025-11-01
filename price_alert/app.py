
import modal
from hello import app, hello, hello_europe
from llama import app, generate
from pricer_ephemeral import app, price

# with app.run():
#     reply=hello.remote()
# print(reply)

# with app.run():
#     reply=hello_europe.remote()
# print(reply)


# with modal.enable_output():
#     reply=generate.remote("What is the capital of France?")
# print(reply)

# with modal.enable_output():
#     with app.run():
#         result=generate.remote("Life is a mystery, everyone must stand alone, I hear")

# print("The results is", result)

# with modal.enable_output():
#     with app.run():
#         result=price.remote("2024 MacBook Pro Laptop with M4 Pro, 14‑core CPU, 20‑core GPU: Built for Apple Intelligence, 16.2-inch Liquid Retina XDR Display, 24GB Unified Memory, 512GB SSD Storage; Space Black")

# print("The results is", result)

from agents.agent import Agent

from agents.specialist_agent import SpecialistAgent
agent = SpecialistAgent()
result = agent.price("2025 MacBook Pro Laptop with M5, 14‑core CPU, 20‑core GPU: Built for Apple Intelligence, 14-inch Liquid Retina XDR Display, 24GB Unified Memory, 1TB SSD Storage; Space Black")
print("The results is", result)