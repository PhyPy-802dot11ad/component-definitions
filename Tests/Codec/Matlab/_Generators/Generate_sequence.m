
rng(1); % Set seed

num_of_sequences = 100;


sequences = randi([0, 1], num_of_sequences, 546);
encoded_sequences = zeros(num_of_sequences, 672);

H_names = [ "H_R_1_2", "H_R_3_4", "H_R_5_8", "H_R_13_16" ];
encoded_sequence_names = [ "encoded_sequences_1_2", "encoded_sequences_3_4", "encoded_sequences_5_8", "encoded_sequences_13_16" ];


        
for H_index = 1:4
    
    H_filename = sprintf("%s.mat", H_names(H_index));
    load(H_filename);
    H = sparse(double(H)); 
    [M,N] = size(H);
    K = N-M;
    
    encoder = comm.LDPCEncoder('ParityCheckMatrix', H);
    
    
    for sequence_idx = 1:num_of_sequences
        
        sequence = sequences( sequence_idx, 1:K );

        encoded_sequence(sequence_idx,:) = encoder(sequence.').';
        
    end
    
    % filename = sprintf( './Datasets/%s.mat', encoded_sequence_names(H_index) );
    filename = sprintf( '../_Input/%s.mat', encoded_sequence_names(H_index) );

    save( filename, 'encoded_sequence' );
    
end


% filename = './Datasets/sequences.mat';
filename = '../_Input/sequences.mat';

save( filename, 'sequences' );
