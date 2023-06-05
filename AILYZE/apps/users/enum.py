from enum import Enum

class Anaylsis(Enum):
    Summarize = "Summarize"
    Ask_a_specific_question = "Ask_a_specific_question"
    Conduct_thematic_analysis = "Conduct_thematic_analysis"
    Identidy_which_document_contain_a_certain_viewpoint = "Identidy_which_document_contain_a_certain_viewpoint"
    Compare_viewpoints_across_documents="Compare_viewpoints_across_documents"


    @classmethod
    def choices(cls):
        return {i.name:i.value for i in Anaylsis}

