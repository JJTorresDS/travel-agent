from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

##res = tavily_search("Best budget hotels in Argentina")
##print(res)

res = search_flights("Plan a 7 day Braszil tripo from Argentina")
print(res)