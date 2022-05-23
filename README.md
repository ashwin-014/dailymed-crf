# dailymed-crf
A Conditional Random Field based NER to find out disease &amp; drug mentions in DailyMed Package Index Data

* This was done in 2018 to learn more about NERs
* I wanted to try out something other than DNNs and hence experimented on CRFs

## Conditional random fields (CRFs)
* CRFs are a type of discriminative undirected probabilistic graphical model
* A CRF can take context into account. To do so, the predictions are modelled as a graphical model, which represents the presence of dependencies between the predictions
* In natural language processing, "linear chain" CRFs are popular, for which each prediction is dependent only on its immediate neighbours