// options
//#define NOVCF                     //if defined, then not use vcf
//#define CHANGE_FORBIDDEN_POLICY   //set policy target of forbidden points to nonzero
//#define FORGOMOCUP                //CPU only and single thread

//�ж�������ѵ�������Ǵ���vcf��
#define TRAINING_DATA_VCF_PROB 0.9

//ѵ�����ݴ��Ž��������ĸ���
#define TRAINING_DATA_FORBIDDEN_FEATURE_PROB 0.5

//#define FORGOMOCUP


#ifndef COMPILE_MAX_BOARD_LEN
#define COMPILE_MAX_BOARD_LEN 15 // Board::MAX_LEN, Normal gomoku/renju games are on 15x15 board
#endif 