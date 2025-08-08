from utils import reformat

from BigScienceP3 import BigScience
from DSFinanceReasoning import FinanceReasoning
from JFFinanceInstruct import FinanceInstruct
from OpenMathReasoning import OpenR1
from SimpleAIH3 import HC3Instruct
from VFinanceInstructReasoning import SynFinanceInstructReason
from ZKSyntheticFinancialDecisionsReasoning import FinancialDecisionsReasoning

from datasets import concatenate_datasets

def main():
    bs=BigScience(2500, 2500)
    op=OpenR1(2500)
    hc=HC3Instruct(count=2500)
    
    fr=FinanceReasoning(400)
    sfr=SynFinanceInstructReason()
    fi=FinanceInstruct(count=5000)
    fdr=FinancialDecisionsReasoning(count=180)

    dataset=concatenate_datasets([bs,op,hc,fr,sfr,fi,fdr])
    dataset.save_to_disk("../mixed_dataset")
    

if __name__ == "__main__":
    main()