@startjson

{
    "database": [
        {
            "authenticationCollection":
            [
                {
                    "userId": "user_unique_id",
                    "username": "username",
                    "email": "email",
                    "password": "encrypted_password",
                    "role": "<Costumer> | <Artist>"
                },
                {
                    "userId": "user_unique_id",
                    "username": "username",
                    "email": "email",
                    "password": "encrypted_password",
                    "role": "<Costumer> | <Artist>"
                },
                {
                    "userId": "user_unique_id",
                    "username": "username",
                    "email": "email",
                    "password": "encrypted_password",
                    "role": "<Costumer> | <Artist>"
                }
            ]
        },
        {
            "allProductsCollection":
            [
                {
                    "productId": "product_unique_id",
                    "productTitle": "product_title",
                    "productDescription": "product_description",
                    "productPrice": "product_price",
                    "favoredByUsers":
                    [
                        "user_unique_id",
                        "user_unique_id",
                        "user_unique_id"
                    ],
                    "ownedByUsers": []
                },
                {
                    "productId": "product_unique_id",
                    "productTitle": "product_title",
                    "productDescription": "product_description",
                    "productPrice": "product_price",
                    "favoredByUsers":
                    [
                        "user_unique_id"
                    ],
                    "ownedByUsers":
                    [
                        "user_unique_id",
                        "user_unique_id",
                        "user_unique_id"
                    ]
                },
                {
                    "productId": "product_unique_id",
                    "productTitle": "product_title",
                    "productDescription": "product_description",
                    "productPrice": "product_price",
                    "favoredByUsers": [],
                    "ownedByUsers":
                    [
                        "user_unique_id"
                    ]
                }
            ]
        },
        {
            "productImages":
            [
                {
                    "productId": "product_unique_id",
                    "productImage": "<product_image>"
                },
                {
                    "productId": "product_unique_id",
                    "productImage": "<product_image>"
                },
                {
                    "productId": "product_unique_id",
                    "productImage": "<product_image>"
                }
            ]
        },
        {
            "productAudioFiles":
            [
                {
                    "productId": "product_unique_id",
                    "productAudios":
                    [
                        {
                            "audioFile": "<audioFile>"
                        },
                        {
                            "audioFile": "<audioFile>"
                        },
                        {
                            "audioFile": "<audioFile>"
                        }
                    ]
                },
                {
                
                    "productId": "product_unique_id",
                    "productAudios":
                    []
                },
                {
                    "productId": "product_unique_id",
                    "productAudios":
                    [
                        {
                            "audioFile": "<audioFile>"
                        }
                    ]
                }
            ]
        }
    ]
}

@endjson
