from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from backend import run_travel_agent

##res = tavily_search("Best budget hotels in Argentina")
##print(res)

##res = search_flights("Plan a 7 day Brazil trip from Argentina")
##print(res)

user_input = input("Enter your travel query: ")
res = run_travel_agent(user_input=user_input, thread_id="test_user")
print(res)