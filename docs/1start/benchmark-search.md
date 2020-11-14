Benchmarking Search Algorithms for Generating NLP Adversarial Examples
=========================================================================


*This documentation page was adapted from Our Paper in [EMNLP BlackNLP](https://arxiv.org/abs/2009.06368).*


### Title: Searching for a Search Method: Benchmarking Search Algorithms for Generating NLP Adversarial Examples


- Abstract:  We study the behavior of several black-box search algorithms used for generating adversarial examples for natural language processing (NLP) tasks. We perform a fine-grained analysis of three elements relevant to search: search algorithm, search space, and search budget. When new search methods are proposed in past work, the attack search space is often modified alongside the search method. Without ablation studies benchmarking the search algorithm change with the search space held constant, an increase in attack success rate could from an improved search method or a less restrictive search space. Additionally, many previous studies fail to properly consider the search algorithms' run-time cost, which is essential for downstream tasks like adversarial training. Our experiments provide a reproducible benchmark of search algorithms across a variety of search spaces and query budgets to guide future research in adversarial NLP. Based on our experiments, we recommend greedy attacks with word importance ranking when under a time constraint or attacking long inputs, and either beam search or particle swarm optimization otherwise. 


###  Citations: 
```
@misc{yoo2020searching,
      title={Searching for a Search Method: Benchmarking Search Algorithms for Generating NLP Adversarial Examples}, 
      author={Jin Yong Yoo and John X. Morris and Eli Lifland and Yanjun Qi},
      year={2020},
      eprint={2009.06368},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



### Our benchmarking results on comparing search methods used in the past attacks. 


![Table1](/_static/imgs/benchmark/search-example.pdf)

![Table1](/_static/imgs/benchmark/search-table2.png)
![Table1](/_static/imgs/benchmark/search-table31.png)
![Table1](/_static/imgs/benchmark/search-table32.png)
![Table1](/_static/imgs/benchmark/search-fig1.png)
![Table1](/_static/imgs/benchmark/search-fig2.png)

![Table1](/_static/imgs/benchmark/search-table1.png)
