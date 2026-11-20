"""Bilingual input fixtures for the original tokenizer, tool tasks, and chat model."""

# Each source passage and its English rendering describe the same content.
TOKENIZER_PASSAGES = (
    ('人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。',
     'Artificial intelligence is a branch of computer science that studies the nature of intelligence and seeks to build machines that respond as people do. Research areas include robotics, speech and image recognition, natural language processing, and expert systems. Since its beginnings, the theory and technology of AI have matured, and its applications continue to expand. Future AI products may become vessels for human knowledge. AI can simulate aspects of consciousness and thinking. Its intelligence differs from human intelligence, but it can think in human-like ways and may even surpass it.'),
    ('星际航行是指在星系内甚至星系间的空间中进行的航行。由于宇宙空间极其广阔，传统的化学火箭动力在恒星间航行时显得力不从心。科学家们提出了多种方案，包括离子推进器、核热火箭、甚至是利用反物质作为能源的设想。此外，曲率驱动和虫洞旅行等科幻概念也在理论物理研究中被反复探讨。尽管目前人类的足迹仅限于月球，但随着核聚变技术和材料科学的突破，前往火星乃至更遥远的太阳系边缘将成为可能。',
     'Interstellar travel means traveling between stars or even between galaxies. Because space is so vast, conventional chemical rockets struggle to cover interstellar distances. Scientists have proposed ion thrusters, nuclear thermal rockets, and even antimatter as an energy source. Theoretical physicists have also explored ideas from science fiction, such as warp drives and wormholes. Humans have only walked as far as the Moon. Advances in nuclear fusion and materials science could make it possible to travel to Mars and eventually to the distant edges of our solar system.'),
    ('Python 是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。It is widely used in data science, machine learning, and web development. 开发者可以利用 NumPy, Pandas, and PyTorch 等库快速构建复杂的应用。学习 Python 的过程非常愉快，因为它的代码读起来就像英语一样。Whether you are a beginner or an expert, Python offers something for everyone.',
     'Python is a high-level programming language known for its concise syntax and powerful ecosystem. It is widely used in data science, machine learning, and web development. Developers can use NumPy, Pandas, and PyTorch to build complex applications quickly. Learning Python is enjoyable because its code reads like English. Whether you are a beginner or an expert, Python offers something for everyone.'),
)

# Pairs used by the agent data and the translation demo (Chinese, English).
TRANSLATION_PAIRS = (
    ('你好世界', 'Hello World'),
    ('早上好', 'Good morning'),
    ('今天天气真好', 'The weather is nice today'),
    ('我喜欢编程', 'I love programming'),
    ('机器学习很有趣', 'Machine learning is interesting'),
    ('生日快乐', 'Happy birthday'),
)

# English city, original city, temperature, English condition, original condition.
WEATHER_EXAMPLES = (
    ('Beijing', '北京', '28°C', 'sunny', '晴'),
    ('Shanghai', '上海', '15°C', 'cloudy', '多云'),
    ('Guangzhou', '广州', '32°C', 'hot and humid', '闷热'),
    ('Shenzhen', '深圳', '30°C', 'sunny', '晴'),
    ('Hangzhou', '杭州', '22°C', 'overcast', '阴'),
    ('Chengdu', '成都', '18°C', 'light rain', '小雨'),
    ('Wuhan', '武汉', '25°C', 'cloudy', '多云'),
    ('Nanjing', '南京', '20°C', 'sunny', '晴'),
    ("Xi'an", '西安', '16°C', 'windy', '大风'),
    ('Chongqing', '重庆', '26°C', 'overcast', '阴'),
    ('Tokyo', 'Tokyo', '12°C', 'sunny', '晴'),
    ('New York', 'New York', '8°C', 'cloudy', '多云'),
    ('London', 'London', '5°C', 'light rain', '小雨'),
    ('Paris', 'Paris', '10°C', 'overcast', '阴'),
    ('Sydney', 'Sydney', '25°C', 'clear', '晴朗'),
)

# Greetings the original checkpoint uses to introduce an untagged final answer.
ANSWER_OPENINGS = (("我是", "I am"), ("您好", "Hello"), ("你好", "Hi"))
