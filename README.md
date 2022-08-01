# SemEval 2019 Unsupervised Lexical Semantic Frame Induction Task
My approach for solving Task 2 from SemEval 2019 competition

## Description
Lexical Frame Induction is defined as the process of grouping verbs and their dependant words in type-feature structures (i.e. frames) in a fully unsupervised manner. The Berkeley FrameNet database is the most well known resource of such typed-feature structures. While lexical frame resources are proved to be helpful (even essential) in a range of NLP tasks and linguistics investigations, building them for new languages and domains is resource intensive and thus expensive. This problem can be alleviated using unsupervised lexical frame induction methods. The goal of this task is to develop a benchmark and allow comparison of unsupervised frame induction systems for building lexical frame resources for verbs and their arguments.

### Subtask 1: Grouping Verbs to Frame Type Clusters
For this subtask, participants are required to assign occurrences of the target verbs to a number of cluster, in such a way that verbs belonging to the same cluster evoke the same frame type. For instance, in the following examples:

a. Trump leads the world, backward.
b. Disrespecting international laws leads to many complications.
c. Rosenzweig heads the climate impacts section at NASA's Goddard Institute.

we expect that the verbs 'to lead' in ex. a and 'to head' in ex. c end up in one cluster (e.g., call it Leadership after FrameNet) whereas 'to lead' in ex. b will end up in another cluster (e.g., call it Cause) in which instances of verbs 'originate', 'produce', an so on (when they are used in the same sense) can be found. As exemplified above, the subtask 1 goes beyond the verb-sense induction task by requiring grouping of synonym, troponym, (even) antonym, ...  senses of verbs together. 

Our annotations for this subtask will be based on FrameNet definitions, where it is covered.

### Subtask 2.1: Clustering arguments of verbs to frame-specific slots
For this subtask, arguments of verbs must be grouped to a number of frame-specific slots similar to FrameNet. That is, we assume argument groupings are specific to frame types and that they are not necessarily shared with other frames. As a result, participating in Subtask 2.1 demands participation in Subtask 1 since evaluations of argument groupings are done per frame cluster. However, one could build frame specific slot-clusters by using a heuristic/assumption such as a frame per verb-form.

### Subtask 2.2: Clustering arguments of verbs to generic roles
In contrast to subtask 2.1, here verb arguments are clustered into a set of generic roles that are defined independently of frame definitions. Hence, this subtask is very similar to unsupervised semantic role induction. Providing frame clustering (i.e., subtask 1) is not mandatory for this subtask and groupings of verbs arguments into latent semantic roles are evaluated disregarding the frames that the verbs belong to.
