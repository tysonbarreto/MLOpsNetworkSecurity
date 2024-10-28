from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str


if __name__=="__main__":
    __all__=["DataIngestionArtifact"]