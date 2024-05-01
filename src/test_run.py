from helper import load_data
from models import prob_ranking, tf_idf_ranking

import time

def main() -> None:
    # load data
    doc_info, inv_idx, vocab = load_data()

    queries = [
        "How to bake a chocolate cake",
        "Best hiking trails near me",
        "What is the capital of Australia?",
        "How to change a flat tire",
        "Healthy breakfast recipes",
        "Current weather forecast in New York City",
        "How to tie a tie",
        "Top 10 action movies of all time",
        "Symptoms of the flu",
        "How to meditate for beginners",
        "Historical events of the 20th century",
        "DIY home decor ideas",
        "Best Italian restaurants in Chicago",
        "How does solar energy work?",
        "How to start a small business",
        "Benefits of yoga",
        "How to grow tomatoes in containers",
        "Famous paintings of the Renaissance",
        "How to train a puppy",
        "How to invest in stocks for beginners",
        "Types of clouds",
        "Romantic weekend getaway ideas",
        "What is the meaning of life?",
        "How to make sushi at home",
        "Healthy meal prep ideas",
        "How to improve memory",
        "Best budget smartphones of 2024",
        "How to write a resume",
        "Causes of insomnia",
        "How to play chess",
        "What is the speed of light?",
        "How to build a website from scratch",
        "DIY car maintenance tips",
        "Best sci-fi books of all time",
        "How to do a French braid",
        "Benefits of mindfulness meditation",
        "How to cook perfect rice",
        "Famous landmarks in Europe",
        "What is global warming?",
        "How to speak Spanish fluently",
        "DIY skincare recipes",
        "Top 10 tourist destinations in Asia",
        "How to start a garden",
        "Health benefits of green tea",
        "How to do a smokey eye makeup",
        "What is the pH scale?",
        "How to make homemade pizza",
        "Best online workout programs",
        "How to build a fire pit",
        "How does the immune system work?",
        "How to knit a scarf",
        "Best budget travel destinations",
        "What is quantum mechanics?",
        "How to make a budget plan",
        "How to fix a leaky faucet",
        "Famous inventions of the 21st century",
        "How to make a compost bin",
        "Benefits of drinking water",
        "How to do a handstand",
        "What is artificial intelligence?",
        "How to organize your closet",
        "Famous female scientists",
        "How to make a paper airplane",
        "Best hiking gear for beginners",
        "How to reduce stress",
        "What is dark matter?",
        "How to make homemade bread",
        "Best time to visit Japan",
        "How to start a YouTube channel",
        "Benefits of intermittent fasting",
        "How to paint a room",
        "Healthy snack ideas for work",
        "How to grow succulents indoors",
        "What is the Fibonacci sequence?",
        "How to build a raised garden bed",
        "Best documentaries on Netflix",
        "How to improve posture",
        "What is the greenhouse effect?",
        "How to do a backflip",
        "How to install a ceiling fan",
        "Famous quotes about success",
        "How to make a resume stand out",
        "Best running shoes for beginners",
        "How to make a vision board",
        "How to repair drywall",
        "What is cryptocurrency?",
        "How to write a cover letter",
        "Best destinations for solo travel",
        "How to make homemade ice cream",
        "What is gene editing?",
        "How to make a fruit smoothie",
        "Best budget laptops for students",
        "How to improve communication skills",
        "How to build a raised vegetable garden",
        "What is the theory of relativity?",
        "How to do a cartwheel",
        "Best podcast for self-improvement",
        "How to tie a bow tie",
        "How to start an online business",
        "Famous speeches in history"
    ]

    print('Testing TF-IDF Model')
    all_query_start = time.time()
    for query in queries:
        # print(f'Query: {query}\n')
        # time_start = time.time()
        rankings = tf_idf_ranking(doc_info, inv_idx, vocab, query, silence=True)
        # time_end = time.time()
        # query_process_time = time_end - time_start
        # print(f'Query process time: {query_process_time:.2f} seconds\n')

    all_query_end = time.time() - all_query_start
    print(f'\tTime to process {len(queries)} queries: {all_query_end:.2f} seconds')
    average_query_time = all_query_end / len(queries)
    print(f'\tAverage query process time: {average_query_time:.2f} seconds')
    print()

    print('Testing Probabilistic Model')
    all_query_start = time.time()
    for query in queries:
        # print(f'Query: {query}\n')
        # time_start = time.time()
        rankings = prob_ranking(doc_info, inv_idx, vocab, query, silence=True)
        # time_end = time.time()
        # query_process_time = time_end - time_start
        # print(f'Query process time: {query_process_time:.2f} seconds\n')

    all_query_end = time.time() - all_query_start
    print(f'\tTime to process {len(queries)} queries: {all_query_end:.2f} seconds')
    average_query_time = all_query_end / len(queries)
    print(f'\tAverage query process time: {average_query_time:.2f} seconds')

    return

if __name__ == "__main__":
    main()
    pass
