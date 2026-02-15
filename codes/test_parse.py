import re
def parse_rationale(response):
    match = re.search(r'Rationale:\s*\[?<*(.*?)>*\]?\s*Rating:', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return None

def parse_rating(response):
    match = re.search(r'Rating:\s*\[?<*(\d+)>*\]?', response)
    if match:
        return int(match.group(1))
    else:
        return None

sample_message =  """Rationale: [<<<This essay presents a clear and well-supported argument that disagrees with the prompt. The response is well-organized with a clear introduction, two distinct body paragraphs, and a conclusion. Each body paragraph focuses on a single reason (strict schedules, lack of opportunity to meet people) and supports it with a detailed and relevant personal example, which is a key strength of the response. The writer demonstrates facility in language use, employing a good range of vocabulary and sentence structures to express ideas clearly. However, the essay is prevented from receiving a top score by a number of noticeable minor errors in grammar, word form, and idiomatic expression (e.g., "where I want go," "go outside in the night," "let me know the ways"). These errors, while frequent, do not obscure the writer's meaning. Overall, the essay fits the description of a Score 4, as it is well-developed and organized but contains some noticeable language errors.>>>]
Rating: [<<<4>>>]"""
print(parse_rationale(sample_message))
print(parse_rating(sample_message))

print("===")
sample_message = """Rationale: The essay addresses the prompt by taking a clear position (disagreeing) and supporting it with reasons and examples. The organization is sound, with an introduction that states the thesis, body paragraphs that attempt to develop distinct points, and a concluding paragraph that reiterates the main argument. The development is generally good; the writer uses examples like the origin of capitalism, the mystery of the pyramids, and conflicting political ideologies to support the idea that facts are foundational.

However, the response is weakened by noticeable and frequent errors in language use. There are errors in word choice ("democratism" for democracy), word form ("origined," "comflict," "mistery"), and grammar ("disagree the statements," "the stock markets was made"). While these errors are persistent, they do not generally interfere with the reader's ability to understand the writer's meaning. The connection of ideas is mostly clear, though the argument in the third paragraph about wars and happiness is less coherent than the others. Overall, the essay addresses the task well and is generally well-developed, fitting the description of a Score 4, which allows for noticeable errors as long as they do not obscure meaning.
Rating: 4"""
print(parse_rationale(sample_message))
print(parse_rating(sample_message))

sample_rubrics = '''```markdown
## Score 5  
An essay at this level:  
- fully addresses all aspects of the topic and task with a clear, well-developed position  
- is thoroughly organized with clear progression and cohesion; ideas are logically connected and fully elaborated through relevant, specific explanations and examples  
- consistently demonstrates facility with language, including syntactic variety, precise and appropriate word choice, and idiomatic expressions  
- contains only minor, infrequent lexical or grammatical errors that do not distract or obscure meaning  

## Score 4  
An essay at this level:  
- addresses all parts of the topic and task clearly and presents a well-developed position, though some points may be less fully elaborated or less specific  
- is generally well organized and coherent, with clear progression of ideas; may contain occasional redundancy, minor digressions, or slightly unclear connections  
- displays good control of language with some variety in sentence structures and vocabulary, despite occasional noticeable minor errors in grammar, word form, or idiomatic usage that do not interfere with meaning  
- examples and explanations are relevant and mostly sufficient but may lack full development or precision  

## Score 3  
An essay at this level:  
- addresses the topic and task but may respond unevenly or incompletely; development of ideas is somewhat limited, vague, or repetitive  
- shows some organization and coherence, but connections between ideas may be inconsistent, unclear, or occasionally obscure meaning  
- demonstrates inconsistent language control with frequent errors in grammar, word choice, or sentence structure that sometimes interfere with clarity or cause minor confusion  
- examples or explanations may be insufficient, general, or only somewhat relevant; vocabulary and syntax are limited in range and sophistication  

## Score 2  
An essay at this level:  
- responds to the topic and task in a limited or minimal way; ideas are underdeveloped and may be off-topic or only loosely connected  
- organization is inadequate, with poor or unclear progression and weak or missing connections between ideas  
- contains frequent and distracting language errors (grammar, word form, sentence structure, vocabulary) that significantly obscure meaning or hinder understanding  
- explanations and examples are inappropriate, insufficient, irrelevant, or largely absent  

## Score 1  
An essay at this level:  
- is seriously underdeveloped or off-topic, showing little or no relevance to the prompt  
- lacks organization to the extent that meaning is obscured or the response is incoherent  
- displays pervasive, severe errors in sentence structure and usage that make comprehension difficult or impossible  
- contains little or no meaningful detail or examples to support ideas  

---

### Additional Guidance for Scoring and Rationales:  
- Focus on the **overall coherence and clarity**: even with frequent errors, if the meaning is generally clear and the ideas are sufficiently developed and organized, the essay should not be rated below 3.  
- Distinguish between **minor errors that do not impede understanding** (Score 4 or 5) and **errors that cause occasional obscurity or confusion** (Score 3).  
- Consider **development and specificity of ideas** carefully: vague or repetitive points with little elaboration align with Score 3; fully developed, specific, and relevant examples and reasoning support Score 4 or 5.  
- Evaluate **organization by the logical flow of ideas and use of transitions**; occasional digressions or unclear connections are tolerated at Score 4 but not beyond.  
- Recognize **language facility** by syntactic variety, appropriate vocabulary, and idiomaticity: limited or repetitive structures and simple vocabulary set Score 3 apart from higher scores.  
- When an essay shows **serious communication problems** that obscure meaning or show minimal relevance or organization, assign Score 2 or below.  

These detailed rubrics and clarifications should help the assistant better differentiate among essays and align scores more closely with the desired outcomes seen in the examples.
####################'''

from inference import parse_new_rubric
print(parse_new_rubric(sample_rubrics))

sample_message = "Rating: Score Point 3  Score Point 3"
def parse_rating_v2(response):
    # Find the substring after 'Rating:' and then extract all numbers from it
    after = re.search(r'Rating:\s*(.*)', response)
    if not after:
        return None
    nums = re.findall(r'\d+', after.group(1))
    if nums:
        return int(nums[-1])  # Return the last matched number after 'Rating:'
    else:
        return None

print(parse_rating_v2(sample_message))
