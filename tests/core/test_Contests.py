import sys
import pytest


from shangrla.core.Audit import Audit, Contest, CVR, Stratum
from shangrla.core.NonnegMean import NonnegMean

###################################################################################################
class TestContests:

    contest_dict = {
                 'AvB': {
                 'id': 'AvB',
                 'name': 'contest_1',
                 'risk_limit': 0.05,
                 'cards': 10**4,
                 'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.PLURALITY,
                 'n_winners': 2,
                 'vote_for': 2,
                 'candidates': ['alice','bob','carol','dave','erin'],
                 'winner': ['alice','bob'],
                 'audit_type': Audit.AUDIT_TYPE.CARD_COMPARISON,
                 'test': NonnegMean.alpha_mart,
                 'use_style': True
                },
                 'CvD': {
                 'id': 'CvD',
                 'name': 'contest_2',
                 'risk_limit': 0.05,
                 'cards': 10**4,
                 'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.SUPERMAJORITY,
                 'n_winners': 1,
                 'candidates': ['alice','bob','carol','dave'],
                 'winner': ['alice'],
                 'audit_type': Audit.AUDIT_TYPE.POLLING,
                 'test': NonnegMean.alpha_mart,
                 'use_style': False
                },
                 'EvF': {
                 'id': 'EvF',
                 'name': 'contest_3',
                 'risk_limit': 0.05,
                 'cards': 10**4,
                 'choice_function': Contest.SOCIAL_CHOICE_FUNCTION.IRV,
                 'n_winners': 1,
                 'candidates': ['alice','bob','carol','dave','elvis'],
                 'winner': ['alice'],
                 'audit_type': Audit.AUDIT_TYPE.CARD_COMPARISON,
                 'test': NonnegMean.alpha_mart,
                 'use_style': False
                }
            }

    def test_contests_from_dict_of_dicts(self):
        atts = ('id','name','risk_limit','cards','choice_function','n_winners','share_to_win','candidates',
                'winner','assertion_file','audit_type','test','use_style')
        contests = Contest.from_dict_of_dicts(self.contest_dict)
        for id, c in contests.items():
            assert c.__dict__.get('id') == id
            for att in atts:
                if att != 'id':
                    assert c.__dict__.get(att) == self.contest_dict[id].get(att)

    def test_tally(self):
        cvr_dict = [{'id': 1, 'votes': {'AvB': {'Alice':True}, 'CvD': {'Candy':True}}},
                    {'id': 2, 'votes': {'AvB': {'Bob':True}, 'CvD': {'Elvis':True, 'Candy':False}}},
                    {'id': 3, 'votes': {'EvF': {'Bob':1, 'Edie':2}, 'CvD': {'Elvis':False, 'Candy':True}}},
                    {'id': 4, 'votes': {'AvB': {'Alice':1}, 'CvD': {'Candy':'yes'}}},
                    {'id': 5, 'votes': {'AvB': {'Bob':True}, 'CvD': {'Elvis':True, 'Candy':False}}},
                    {'id': 6, 'votes': {'EvF': {'Bob':1, 'Edie':2, 'Dan':[3,4], 'Elvis':4}, 'CvD': {'Elvis':False, 'Candy':True}}},
                    {'id': 7, 'votes': {'AvB': {'Alice':2}, 'CvD': {'Elvis':False, 'Candy':True}}}
                   ]
        cvr_list = CVR.from_dict(cvr_dict)
        contests = Contest.from_dict_of_dicts(self.contest_dict)
        Contest.tally(contests, cvr_list)
        assert contests['AvB'].tally == {'Alice': 3, 'Bob': 2}
        assert contests['CvD'].tally == {'Candy': 5, 'Elvis': 2}
        assert contests['EvF'].tally is None 
        # TO DO: assert that this raises a warning about contest EvF

    def test_validate_vote(self):
        contests = Contest.from_dict_of_dicts(self.contest_dict)
        cvr_dict = [{'id': 1, 'votes': {'AvB': {'Alice':True}, 'CvD': {'Candy':True}}},
                    {'id': 2, 'votes': {'AvB': {'Alice':True, 'Bob':True, 'Candy':True}, 'CvD': {'Elvis':True, 'Candy':False}}},
                    {'id': 3, 'votes': {'AvB': {'Alice':True, 'Bob':True, 'Candy':True}, 'CvD': {'Elvis':True, 'Candy':True}}},
                    {'id': 4, 'votes': {'AvB': {'Alice':True, 'Bob':False, 'Candy':False}, 'CvD': {'Elvis':True, 'Candy':False}}},
                    {'id': 5, 'votes': {'EvF': {'Bob':1, 'Candy':2, 'Dan':[3,4], 'Elvis':4}, 'CvD': {'Elvis':False, 'Candy':True}}},
                    {'id': 6, 'votes': {'EvF': {'Bob':1, 'Candy':2, 'Dan':[3,4], 'Elvis':3}, 'CvD': {'Elvis':False, 'Candy':True}}},
                    {'id': 7, 'votes': {'EvF': {'Bob':2, 'Candy':4, 'Dan':5, 'Elvis':6}, 'CvD': {'Elvis':False, 'Candy':True}}},
                    {'id': 8, 'votes': {'EvF': {'Bob':2, 'Candy':4, 'Dan':4, 'Elvis':5}, 'CvD': {'Elvis':False, 'Candy':True}}},
                   ]
        cvr_list = CVR.from_dict(cvr_dict)
        for con in contests.values():
            print(f'{str(con)=}')
        for con in contests.values():
            assert not cvr_list[0].contest_validated(con.id)
        for cvr in cvr_list:
            for con in contests.values():
                con.validate_vote(cvr)
        for con in contests.values():
            for cvr in cvr_list:
             assert cvr.contest_validated(con.id)
        assert cvr_list[0].votes['AvB'] == {'Alice':True} and cvr_list[0].votes['CvD'] == {'Candy':True}
        assert (cvr_list[1].votes['AvB'] == {'Alice':False, 'Bob':False, 'Candy':False} 
                and cvr_list[1].votes['CvD'] == {'Elvis':True, 'Candy':False})
        assert (cvr_list[2].votes['AvB'] == {'Alice':False, 'Bob':False, 'Candy':False} 
                and cvr_list[2].votes['CvD'] == {'Candy':False, 'Elvis':False})
        assert (cvr_list[3].votes['AvB'] == {'Alice':True, 'Bob':False, 'Candy':False}
                and cvr_list[3].votes['CvD']== {'Elvis':True, 'Candy':False})
        print(f'{cvr_list[4].votes=}')
        print(f'{contests['EvF'].choice_function=}')
        assert cvr_list[4].votes['EvF'] == {'Bob':1, 'Candy':2, 'Dan':3}
        assert cvr_list[5].votes['EvF'] == {'Bob':1, 'Candy':2}
        assert cvr_list[6].votes['EvF'] == {'Bob':1, 'Candy':2, 'Dan':3, 'Elvis':4}
        assert cvr_list[7].votes['EvF'] == {'Bob':1}
        

##########################################################################################
if __name__ == "__main__":
    sys.exit(pytest.main(["-qq"], plugins=None))
