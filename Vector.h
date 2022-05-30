#pragma once

/**
 * @file Vector.h
 *
 * @brief Implantation de vecteurs simples
 *
 * Nom: William Lebel
 * Email : william.lebel.1@ens.etsmtl.ca
 *
 */

#include <cmath>
#include "MatrixBase.h"
#include "Assert.h"

namespace gti320
{
	/**
	 * Classe vecteur générique.
	 *
	 * Cette classe réutilise la classe `MatrixBase` et ses spécialisations de
	 * templates pour les manipulation bas niveau.
	 */
	template <typename Scalar = double, int Rows = Dynamic>
	class Vector : public MatrixBase<Scalar, Rows, 1>
	{
	public:
		/**
		 * Constructeur par défaut
		 */
		Vector() : MatrixBase<Scalar, Rows, 1>()
		{
		}

		/**
		 * Constructeur avec une liste d'initialistation
		 *
		 * Ce constructeur est plus lent que les autres et ne devraient être utilisé qu'en cas de nécessité ou de test.
		 */
		Vector(const std::initializer_list<Scalar>& initializerList) : MatrixBase<Scalar, Rows, 1>()
		{
			ASSERT(Rows == Dynamic || this->size() == initializerList.size(), "Attempting to create a vector with an initializer list of an invalid size")

				this->resize(initializerList.size());

			auto i = 0;
			for (auto element = initializerList.begin(); element != initializerList.end(); ++element)
			{
				(*this)(i) = *element;
				++i;
			}
		}

		/**
		 * Contructeur à partir d'un taille (m_rows).
		 */
		explicit Vector(int rows) : MatrixBase<Scalar, Rows, 1>(rows, 1)
		{
		}

		/**
		 * Constructeur de copie
		 */
		Vector(const Vector& other) : MatrixBase<Scalar, Rows, 1>(other)
		{
		}

		/**
		 * Destructeur
		 */
		~Vector()
		{
		}

		/**
		 * Opérateur de copie
		 */
		Vector& operator=(const Vector& other)
		{
			this->resize(other.size());
			memcpy(this->m_storage.data(), other.m_storage.data(), sizeof(Scalar) * other.size());

			return *this;
		}

		/**
		 * Accesseur à une entrée du vecteur (lecture seule)
		 */
		Scalar operator()(int i) const
		{
			return this->m_storage[i];
		}

		/**
		 * Accesseur à une entrée du vecteur (lecture et écriture)
		 */
		Scalar& operator()(int i)
		{
			return this->m_storage[i];
		}

		/**
		 * Modifie le nombre de lignes du vecteur
		 */
		void resize(int rows)
		{
			MatrixBase<Scalar, Rows, 1>::resize(rows, 1);
		}

		/**
		 * Produit scalaire de *this et other.
		 */
		inline Scalar dot(const Vector& other) const
		{
			ASSERTF(this->size() == other.size(), "Attempting to compute the dot product of vectors of two different dimensions (%d and %d)", this->size(), other.size());

			Scalar dotProduct = 0;
			for (auto i = 0; i < this->size(); ++i)
			{
				dotProduct += (*this)(i) * other(i);
			}

			return dotProduct;
		}

		/**
		 * Retourne la norme euclidienne du vecteur
		 */
		inline Scalar norm() const
		{
			return sqrt(this->squaredNorm());
		}

		/**
		 * Retourne la norme euclidienne à la 2 du vecteur
		 */
		inline Scalar squaredNorm() const
		{
			Scalar squaredNorm = 0;
			for (auto i = 0; i < this->size(); ++i)
			{
				squaredNorm += (*this)(i) * (*this)(i);
			}

			return squaredNorm;
		}

		// Ces fonctions retourne les coordonnées 0, 1, 2 et 3 de vecteurs en lecture et en écriture de manière plus intuitive
		inline Scalar x() const { return (*this)(0); }
		inline Scalar& x() { return (*this)(0); }
		inline Scalar y() const { return (*this)(1); }
		inline Scalar& y() { return (*this)(1); }
		inline Scalar z() const { return (*this)(2); }
		inline Scalar& z() { return (*this)(2); }
		inline Scalar w() const { return (*this)(3); }
		inline Scalar& w() { return (*this)(3); }
	};
}
