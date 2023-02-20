import os
from typing import List

import numpy as np
import torch
from transformers import GPT2LMHeadModel
from transformers import PreTrainedTokenizerFast

from sample import midi_to_text


PPLM_BOW = 1
BOW_AFFECT = 4

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3

VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}


def run_pplm_example(pretrained_model="gpt2-medium",
                     priming_sample="",
                     affect_weight=0.2,
                     knob=None,
                     uncond=False,
                     num_samples=1,
                     bag_of_words=None,
                     bag_of_words_affect=None,
                     discrim_weights=None,
                     discrim_meta=None,
                     class_label=-1,
                     length=100,
                     stepsize=0.02,
                     temperature=1.0,
                     top_k=10,
                     sample=True,
                     num_iterations=3,
                     grad_length=10000,
                     horizon_length=1,
                     window_length=0,
                     decay=False,
                     gamma=1.5,
                     gm_scale=0.9,
                     kl_scale=0.01,
                     seed=0,
                     no_cuda=False,
                     colorama=False,
                     verbosity='regular',
                     n_bar_window=2):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # load pretrained model and tokenizer
    model, tokenizer = load_model(n_bar_window)
    model.to(device)
    model.eval()

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    input_ids = tokenizer.encode(priming_sample, return_tensors="pt")

    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
        model=model,
        tokenizer=tokenizer,
        affect_weight=affect_weight,
        knob=knob,
        context=input_ids,
        device=device,
        num_samples=num_samples,
        bag_of_words=bag_of_words,
        bag_of_words_affect=bag_of_words_affect,
        class_label=class_label,
        length=length,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        window_length=window_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        verbosity_level=verbosity_level
    )


def load_model(n_bar_window):
    model_file = f'gpt2model_{n_bar_window}_bars'
    tokenizer_path = os.path.join(model_file, "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model_path = os.path.join(model_file, "best_model")
    model = GPT2LMHeadModel.from_pretrained(model_path)

    return model, tokenizer


def full_text_generation(
        model,
        tokenizer,
        affect_weight=0.2,
        knob=None,
        context=None,
        num_samples=1,
        device="cuda",
        bag_of_words=None,
        bag_of_words_affect=None,
        class_label=None,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        **kwargs
):
    bow_indices = []
    bow_indices_affect = []

    if bag_of_words_affect:
        affect_words, affect_int = get_affect_words_and_int(bag_of_words_affect)
        bow_indices_affect.append([tokenizer.encode(word.strip(), add_prefix_space=True, add_special_tokens=False) for word in affect_words])

    loss_type = PPLM_BOW
    if bag_of_words_affect:
      loss_type = BOW_AFFECT

    # unpert_gen_tok_text, _, _ = generate_text_pplm(
    #     model=model,
    #     tokenizer=tokenizer,
    #     context=context,
    #     device=device,
    #     length=length,
    #     sample=sample,
    #     perturb=False,
    #     verbosity_level=verbosity_level
    # )
    #
    # if device == 'cuda':
    #     torch.cuda.empty_cache()
    #
    # pert_gen_tok_texts = []
    # discrim_losses = []
    # losses_in_time = []
    # print("After Perturbation")
    # for i in range(num_samples):
    #     pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
    #         model=model,
    #         tokenizer=tokenizer,
    #         affect_weight=affect_weight,
    #         context=context,
    #         device=device,
    #         perturb=True,
    #         bow_indices=bow_indices,
    #         bow_indices_affect=bow_indices_affect,
    #         affect_int = affect_int,
    #         knob = knob,
    #         classifier=classifier,
    #         class_label=class_id,
    #         loss_type=loss_type,
    #         length=length,
    #         stepsize=stepsize,
    #         temperature=temperature,
    #         top_k=top_k,
    #         sample=sample,
    #         num_iterations=num_iterations,
    #         grad_length=grad_length,
    #         horizon_length=horizon_length,
    #         window_length=window_length,
    #         decay=decay,
    #         gamma=gamma,
    #         gm_scale=gm_scale,
    #         kl_scale=kl_scale,
    #         verbosity_level=verbosity_level
    #     )
    #     pert_gen_tok_texts.append(pert_gen_tok_text)
    #     if classifier is not None:
    #         discrim_losses.append(discrim_loss.data.cpu().numpy())
    #     losses_in_time.append(loss_in_time)
    #
    # if device == 'cuda':
    #     torch.cuda.empty_cache()
    #
    # return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
    return None, None, None, None


def get_affect_words_and_int(affect_class):
    """
    Нужно получить такты, соответствующие эмоциям
    :param affect_class:
    :return:
    """
    emotions = ""
    filepath = cached_path(emotions)

    with open(filepath, "r") as f:
        words = f.read().strip().split("\n")[1:]

    words = [w.split("\t") for w in words]

    return [w[0] for w in words if w[1] == affect_class], [float(w[-1]) for w in words if w[1] == affect_class]


def generate(priming_sample):
    topics = [
        'legal']  # ,'military','monsters','politics','positive_words', 'religion', 'science','space','technology']
    affects = ['fear']  # , 'anger', 'sadness'] #'fear',
    knob_vals = [0.8]  # ,0.5,0.7,1]

    for topic in topics:
        for affect in affects:
            for knob in knob_vals:
                print("topic:", topic, ", affect:", affect, ", knob is:", knob)
                run_pplm_example(
                    affect_weight=1,  # it is the convergence rate of affect loss, don't change it :-p
                    knob=knob,  # 0-1, play with it as much as you want
                    priming_sample=priming_sample,
                    num_samples=1,
                    bag_of_words=topic,
                    bag_of_words_affect=affect,
                    length=50,
                    stepsize=0.005,  # topic, affect convergence rate
                    sample=True,
                    num_iterations=10,
                    window_length=5,
                    gamma=1.5,
                    gm_scale=0.95,
                    kl_scale=0.01,
                    verbosity='quiet'
                )


if __name__ == '__main__':
    filename = os.path.join('data', 'jingle_bells.mid')
    split_path = os.path.split(filename)
    priming_txt = f"{split_path[1].split('.')[0]}.txt"
    priming_txt = os.path.join('text_representations', priming_txt)
    midi_to_text(filename, priming_txt)

    with open(priming_txt, 'r') as hfile:
        sample = hfile.read()

    generate(sample)
