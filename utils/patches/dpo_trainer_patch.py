def patch_dpo_trainer(DPOTrainer):
    def dummy_tokenize_row(self, feature):
        return feature

    DPOTrainer.tokenize_row = dummy_tokenize_row
