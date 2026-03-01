# SmartFridgeAssistant4-web
Version 4. More smooth, more seamless, the better.

The Smart Fridge Assistant Website (Jarvis) version 4 enable user to check their fridge content on the go, anywhere, anytime, as long as connected to the internet when using our product (Javis AI). 
Version 4 update: 
-Automatically detect Arduino Uno USB port.
-Shows indication on the Arduino Uno display
-Fix bugs
-Added Jarvis Talk back feature

Website: https://theern06.github.io/SmartFridgeAssistant4-web/
User have to log in before accessing to the Jarvis website. 

The version 4 website consists of 5 main page which are:
Inventory page,
Summary (Waste Food Data),
Guide,
About Us,
Jarvis,


Page 1: Inventory Page 
The inventory page shows the summary data (food wasted) and a list of inventory on the user's fridge in a table form. The table is linked to a google sheet where the google sheet saved all the food content. The google sheet have an extentsion called apps script. In the apps script, a code file named code.gs have a unquie URL (inventory.html, line 151) which enable google sheet to update from Jarvis and to display at Jarvis Website. User able to search the specific food, sort the items as long as the website is connect to the internent.

Note that the Summary sentence takes a few seconds to load.

Page 2: Summary  
Summary page shows the summary daya (food wasted) and all the foods or items that are already expired in terms of last 7 days (Weekly Waste Report) and last 30 days (Monthly Waste Report) in the form of tables. The count is based on the food quantity e.g 2 apples and 1 milk expired 3 days ago and 34 days ago: 2 foods wasted last 7 days and 3 foods wasted last 30 days.

The following shows the typical shelf life that Jarvis can understand. (In Terms of Days)

SHELF_LIFE_RULES = { "protein": 4, "seafood": 7, "dairy": 10, "vegetables": 7, "fruits": 7, "grains": 14,
"pantry": 365, "processed": 7, "non-food": 0 }

Note that the Summary sentence takes a few seconds to load.

Page 3: Guide Guide page shows the system overview, hardware & interaction guide and website navigation guide to the user so that he or she can use it seamlessly.

Page 4: About Us About us page shows some information about this project and some of the models specifications used.

Page 5: Jarvis Page
Before entering the Jarvis page, user are require to connect the same network as the PC network. Upon entering the Jarvis Page, the website will ask the user to enter the IP address as shown in the Arduino LED 16x2 display (when restart the PC). Once sucessfully entered, user able to chat via text or voice. All the talking feature is avaliable same as directly talked to the fridge (PC). 

Note that the website will automatically open a new tab and automatically close after user clikced "Go back".


