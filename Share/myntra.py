import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv('myntra_train_dataset.csv')
data=data.fillna(method='ffill')
kind =['Men', 'Women', 'Girls', 'Unisex', 'Boys']
sub =['Graphic', 'Biker', 'Striped', 'Colourblocked', 'Tie and Dye',
       'Solid', 'Typography', 'Geometric', 'Camouflage', 'Abstract',
       'Tribal', 'People and Places', 'Conversational', 'Sports', 'Floral',
       'Humour and Comic', 'Music', 'Checked', 'Self Design', 'Varsity',
       'Sports and Team Jersey', 'Polka Dots', 'Superhero',
       'Horizontal Stripes']

brand =['Roadster', 'Roadster Fast and Furious', 'Riverstone', 'Riot',
       'Rinascimento', 'Rigo', 'Rider Republic', 'Richlook', 'Rham',
       'Restless', 'Republic of Spiel', 'Replay', 'Renka', 'Reebok',
       'Reebok Classic', 'Redwolf', 'Red Tape', 'Real Madrid', 'Raymond',
       'Rattrap', 'Rajesh Pratap Singh', 'RUFF', 'RSVP Cross', 'ROUTE 66',
       'ROOTSTOCK', 'RODID', 'ROCX', 'RIG', 'RIDRESS', 'REVO', 'RDSTR',
       'R&C', 'Quiksilver', 'QUIZ', 'Push & Pull', 'Purple Feather',
       'Pure Play', 'Punkster', 'Punisher', 'Puma', 'Proline',
       'Proline Active', 'Probase', 'Private Lives', 'PrettySecrets',
       'Posterboy', 'PostFold', 'Polo Ralph Lauren', 'Pokemon', 'Pluto',
       'Pluie', 'Playboy', 'Platinum League', 'Planet Superheroes',
       'Pirates of the Caribbean', 'Pique Republic', 'Pink Floyd',
       'Phosphorus', 'Peter England', 'Peter England Elite',
       'Peter England Casuals', 'Pepito', 'Pepe Jeans', 'People',
       'Peanuts', 'Peach Boy', 'Parx', 'Park Avenue', 'Park Avenue Woman',
       'Palm Tree', 'Paani Puri', 'PURYS', 'PUNK', 'POPPERS by Pantaloons',
       'PERF', 'Oxolloxo', 'Oshea', 'Original Penguin', 'Orange Valley',
       'Oner', 'Okane', 'Obidos', 'OVS', 'OPt', 'ONN', 'ONLY',
       'ONLY & SONS', 'ODAKA', 'Nuteez', 'Numero Uno', 'NuBella',
       'Northern Lights', 'North Coast', 'Nord51', 'Noi', 'Noble Faith',
       'Nirvana', 'Nino Bambino', 'Ninja Turtle', 'Nineteen', 'Nike',
       'New Look', 'New Balance', 'Nautica', 'Nauti Nati', 'NUSH',
       'NU ECO', 'NO.99', 'NEVA', 'NBA', 'Mysin', 'Myntra',
       'My Little Lambs', 'My Lil Berry', 'Music', 'Mufti', 'Ms.Taken',
       'Mr. Men', 'Mr. Men Little Miss', 'Mr Bowerbird', 'Motu Patlu',
       'Mossimo', 'Monteil & Munero', 'Monte Carlo', 'Moda Rapido',
       'Moda Rapido Star Wars', 'Moda Rapido Marvel', 'Moda Rapido Disney',
       'Mizuno', 'Miss Grace', 'Miss Chick', 'Miss Chase', 'Miss Alibi',
       'Miso', 'Minnie', 'Minnie Mouse', 'Minions',
       'Minions by Kook N Keech', 'Mickey', 'Mickey & Friends',
       'Metersbonwe', 'Meish', 'Meiro', 'Meira', 'Meee', 'Mast & Harbour',
       'Maserati', 'Masculino Latino', 'Marvel', 'Marvel Spiderman',
       'Marvel Comics', 'Marvel Avengers', 'Martini', 'Marks & Spencer',
       'Marie Claire', 'Manola', 'Mango Kids', 'Manchester City FC',
       'Man of Steel', 'Maine', 'Madlove', 'Madame', 'MTV', 'MR BUTTON',
       'MOXI', 'MIWAY', 'MINI KLUB', 'MELTIN', 'MARD', 'MANGO',
       'Love Genration', 'Louis Philippe', 'Louis Philippe Sport',
       'Louis Philippe Jeans', 'Lotto', 'Looney Tunes', 'Lonsdale',
       'London Bridge', 'London Bee', 'Loco En Cabeza',
       'Liverpool Football Club UK', 'Liverpool FC', 'Lilliput',
       'LilPicks', 'Lil Tomatoes', 'Lil Orchids', 'Levis', 'Leo',
       'Leo Sansini', 'Lee', 'Lee Cooper', 'Le Bison', 'Lawman pg3',
       'Latin Quarters', 'Lacoste', 'Label Ritu Kumar', 'Laabha',
       'LOVE GEN', 'LOCOMOTIVE', 'LINKIN PARK', 'L.A. SEVEN', 'Kylo Ren',
       'Kung Fu Panda', 'Kulture Shop', 'Kraus Jeans', 'Kook N Keech',
       'Kook N Keech Star Wars', 'Kook N Keech Pokemon',
       'Kook N Keech Music', 'Kook N Keech Marvel',
       'Kook N Keech Garfield', 'Kook N Keech Disney',
       'Kook N Keech Archie', 'Kiwi', 'Killer', 'Kids Ville', 'Kazo',
       'Karrimor', 'Karma', 'Kappa', 'Kapapai', 'Kanvin', 'Kangol',
       'KULTPRIT', 'KOLKATA KNIGHT RIDERS', 'KAARYAH', 'Justice League',
       'Justanned', 'Juniors by Lifestyle', 'Joshua Tree', 'Joker',
       'Johnny Bravo', 'John Pride', 'John Players', 'John Miller',
       'John Miller Hangout', 'Jogur', 'Jockey', 'Jn Joy', 'Jimi Hendrix',
       'Jhonny Bravo', 'Jeep', 'Jealous 21', 'Jazzup', 'Jack & Jones',
       'JUSTICE', 'JUNAROSE', 'JM Sport', 'JAINISH', 'Izinc', 'Iron Man',
       'Integriti', 'Inmark', 'Inego', 'Indigo Nation',
       'Indigo Nation Street', 'Indian Terrain', 'Incynk', 'Incult',
       'Imagica', 'IZOD', 'INVICTUS', 'INDICODE', 'IDK', 'IDENTITI', 'ICC',
       'I AM FOR YOU', 'Hypernation', 'Hulk', 'Huetrap', 'Hubberholme',
       'House of Chase', 'Hot Wheels', 'Hook & Eye', 'Honey by Pantaloons',
       'Hols', 'Henry and Smith', 'Hello Kitty', 'Heart 2 Heart',
       'Harvard', 'Harry Potter', 'Harley-Davidson', 'Happy Hippie',
       'Hangup', 'Hanes', 'HUSTLE', 'HUNGOVER', 'HRX by Hrithik Roshan',
       'HIGHLANDER', 'HERE&NOW', 'HALO 5', 'H.E. By Mango', 'Guns & Roses',
       'Guardians of the Galaxy', 'Greenwich United Polo Club',
       'Green Day', 'Goofy', 'Go-Art', 'Gmcks', 'Globus', 'Globe',
       'Globalite', 'Global Desi', 'Gini and Jony', 'Ginger by Lifestyle',
       'Gesture Jeans', 'Garfield', 'Garcon',
       'Game of Thrones by Kook N Keech', 'Game Of Thrones', 'Gabi',
       'GUESS', 'GRITSTONES', 'GRAIN', 'GOAT', 'GKIDZ', 'GAS', 'GANT',
       'G-STAR RAW', 'Fusion Beats', 'Fugue', 'Frozen',
       'French Connection', 'Free Authority', 'Free & Young',
       'Franco Leone', 'Fox', 'Four One Oh', 'Fort Collins', 'Forever New',
       'Foreign Culture', 'Force NXT', 'Force Go Wear', 'Forca',
       'Flying Machine', 'Firetrap', 'Filmwear', 'FiTZ', 'Ferrari',
       'Feneto', 'Femella', 'Fantastic Beasts',
       'Fantastic Beasts by Kook N Keech', 'Family Guy',
       'Fame Forever by Lifestyle', 'Facit', 'Fabindia', 'FabAlley',
       'FabAlley Curve', 'Fab Deanta', 'FUGAZEE', 'FS Mini Klub', 'FROST',
       'FRITZBERG', 'FREECULTR', 'FREECULTR Express',
       'FOW Friends of Wild', 'FOREVER 21', 'FILA', 'FIFTY TWO',
       'FIFA U-17 WC', 'FCUK', 'FC Barcelona', 'Evoke 1899', 'Everlast',
       'Ethane', 'Espresso', 'Encrypt', 'Emporio Armani', 'Eminem', 'Elle',
       'Elle Kids', 'Eimoie', 'Ed Hardy', 'Ecko Unltd', 'Easies',
       'EVAH LONDON', 'ESPRIT']

rang =['Black', 'Beige', 'White', 'Red', 'Off White', 'Maroon', 'Grey',
       'Grey Melange', 'Cream', 'Coral', 'Blue', 'Teal', 'Olive',
       'Navy Blue', 'Magenta', 'Khaki', 'Coffee Brown', 'Charcoal',
       'Burgundy', 'Brown', 'Yellow', 'Purple', 'Orange', 'Green', 'Multi',
       'Pink', 'Lime Green', 'Turquoise Blue', 'Skin', 'Mustard',
       'Sea Green', 'Peach', 'Mauve', 'Lavender', 'Fluorescent Green',
       'Taupe', 'Rust', 'Rose', 'Bronze', 'Nude', 'Tan', 'Silver', 'Gold',
       'Steel']

data =data[['Brand', 'Gender', 'Color','Sub_category']]

for i in range(0,len(data)):
    for j in range(0,len(sub)):
        if(data.Sub_category[i]==sub[j]):
            data.Sub_category[i]=j
            break;

for i in range(0,len(data)):
    for j in range(0,len(brand)):
        if(data.Brand[i]==brand[j]):
            data.Brand[i]=j+25
            break;
for i in range(0,len(data)):
    for j in range(0,len(rang)):
        if(data.Color[i]==rang[j]):
            data.Color[i]=j+403
            break;
for i in range(0,len(data)):
    for j in range(0,len(kind)):
        if(data.Gender[i]==kind[j]):
            data.Gender[i]=j+448
            break;
    

print(data.head(5))
